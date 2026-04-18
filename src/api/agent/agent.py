from typing import TypedDict, Sequence, Annotated
from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langgraph.graph.message import add_messages

from src.config import config
from src.api.agent.tools import *
from src.logging import logger

class SquadState(TypedDict):
    user_input: Annotated[HumanMessage, "User's query input."]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    architect_out: Annotated[AIMessage, "Decomposed user's query by architect AI agent."]
    coder_output: Annotated[AIMessage, "AI agent coder's output on user's query decomposed by Architect AI agent."]
    tester_feedback: Annotated[Sequence[AIMessage], add_messages]

class AgentSquad():
    def __init__(self):

        self.tools = [python_repl_tool]

        openai_model = "gpt-4o-mini"
        self.architect_llm = ChatOpenAI(openai_model, 
                                    temperature=0.5,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.architect_prompt = ChatPromptTemplate.from_messages([
            ("system", ("You're helpful architect assistant in coding AI agent squad",
                        "Your responsibility in the first place is to take users' query ",
                        "and decompose it into meaningful technical tasks that will be passed to",
                        " the Coding AI agent that must develop fully-working solution without a single mistake."
                        )),
            ("human", "{input}"),
        ])
        self.architect = (
            {
                "input": lambda x: x["input"]
            }
            | self.self.architect_prompt
            | self.architect_llm
        )
        self.coder_llm = ChatOpenAI(openai_model, 
                                    temperature=0.0,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.code_prompt = ChatPromptTemplate.from_messages([
            ("system", ("You're a helpful coder in coding AI agent squad",
                        "Your responsibility in the first place is to take decomposed task from AI assistant architect ",
                        "and implement all technical tasks with maximum precision and reliability.",
                        "you must develop a fully working solution that is 100 percent reliable and working. ",
                        "results of your work would be reviewed by testing system and feedback would be provided ",
                        "in tester_feedback variable. Don't let the user and your team down you're the heart of the system."
                        )),
            ("human", "{architect_out}"),
            MessagesPlaceholder(variable_name="tester_feedback")
        ])
        self.coder = (
            {
                "architect_out": lambda x: x["architect_out"],
                "tester_feedback": lambda x: x.get("tester_feedback", []),
            }
            | self.code_prompt
            | self.coder_llm
        )
        self.tester_llm = ChatOpenAI(openai_model, 
                                    temperature=0.0,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.tester_prompt = ChatPromptTemplate.from_messages([
            ("system", ("You're a helpful tester system in huge coding AI agent squad",
                        "You're playing one of the most important roles in whole system",
                        "Your responsibility here is to take the output code from the coding agent and test it ",
                        "you must use provided REPL tool to execute code then observe the result ",
                        "and give proper feedback about that code.",
                        "YOUR MAIN GOAL IS NOT TO PASS CODE THAT DOESN'T WORK. IF code provided doesn't seem to be working",
                        "you write a report and feedback about code in specific 'tester_feedback' variable. That feedback ",
                        "would be added to prompt for coding AI Agent in next coding iteration.",
                        "If code works and tests are passed. YOU return just one word in strict format: 'Passed'",
                        "that output would be used as condition for next coding agent iteration.",
                        "DON'T LET THE USER AND YOUR TEAM DOWN!"))
            ("human", "coder_output")
        ])
        self.tester = (
            {
                "coder_output": lambda x: x["coder_output"]
            }
            | self.tester_prompt
            | self.tester_llm.bind_tools(tools=self.tools, tool_choice="any")
        )
        self.agent = self._compose_graph(show_image=True)


    def _decomposition_node(self, state: SquadState) -> SquadState:
        response = self.architect.invoke({"input": state["user_input"]})
        state["messages"].append(state["user_input"])
        state["architect_out"] = response
        logger.info(f"[USER]: {state['user_input']}")
        logger.info(f"[AI_ARCHITECT]: {response}")
        return state
    
    def _coding_node(self, state: SquadState) -> SquadState:
        response = self.coder.invoke({"architect_out": state["architect_out"], "tester_feedback": state["tester_feedback"]})
        state["coder_output"] = response
        logger.info(f"[AI_CODER]: {response}")
        return state
    
    def _testing_node(self, state: SquadState) -> SquadState:
        response = self.tester.invoke({"coder_output": state["coder_output"]})
        state["tester_feedback"].append(response)
        state["tester_feedback"] = state["tester_feedback"][:4]
        logger.info(f"[AI_TESTER]: {state["tester_feedback"]}")
        return state
    
    def _if_errors_path(state: SquadState):
        logger.info("[_if_errors_path] state['tester_feedback'][-1]: ", state["tester_feedback"][-1])
        if state["tester_feedback"][-1] == "Passed":
            return "no_errors"
        return "errors"

    def _compose_graph(self, show_image: bool = True):
        graph = StateGraph(AgentSquad)

        graph.add_node("decomposition", self._decomposition_node)
        graph.add_node("coding", self._coding_node)
        graph.add_node("testing", self._testing_node)
        graph.add_node("tools", ToolNode(tools = self.tools))

        graph.set_entry_point("decomposition")
        
        graph.add_edge("decomposition", "coding")
        graph.add_edge("coding", "testing")
        graph.add_conditional_edges(
            source="testing",
            path=self._if_errors_path,
            path_map={
                "errors": "coding",
                "no_errors": END
            }
        )
        
        agent = graph.compile()

        if show_image:
            display(Image(agent.get_graph().draw_mermaid_png()))

        return agent

    def invoke(self, input_data: dict, verbose: bool = False) -> dict:
        init_state = SquadState(user_input=input_data["input"],
                                messages=[],
                                coder_output=None,
                                architect_out=None,
                                tester_feedback=[]
                                )
        response = self.agent.invoke(init_state)
        return response
        