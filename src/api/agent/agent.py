from typing import TypedDict, Sequence, Annotated
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src.config import config
from src.api.agent.tools import *
from src.logging import logger

import json
import asyncio

class SquadState(TypedDict):
    user_input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    architect_out: str
    coder_output: str
    tester_feedback: Annotated[Sequence[AIMessage], add_messages]
    recode_attempts: int

class AgentSquad():
    def __init__(self):

        self.tools = [python_repl_tool]

        openai_model = "gpt-4o-mini"
        self.architect_llm = ChatOpenAI(name=openai_model, 
                                    temperature=0.5,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.architect_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a helpful architect assistant in a coding AI agent squad.
            Your responsibility is to take users' query and decompose it into meaningful 
            technical tasks for a Python Coding AI agent that must develop a fully-working solution."""),
            ("human", "{input}"),
        ])
        self.architect = (
            {
                "input": lambda x: x["input"]
            }
            | self.architect_prompt
            | self.architect_llm
        )
        self.coder_llm = ChatOpenAI(name=openai_model, 
                                    temperature=0.0,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.code_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a helpful coder in an AI squad.
                        Your task is to take the Architect's plan and implement it in Python.
                        - Provide ONLY the Python code.
                        - Ensure the code is self-contained and runnable.
                        - If you receive 'tester_feedback', use it to fix the previous version of your code.
                        - Do not explain yourself. Just provide the code."""),
            ("human", "Architect Plan: {architect_out}"),
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
        self.tester_llm = ChatOpenAI(name=openai_model, 
                                    temperature=0.0,
                                    api_key=config["OPENAI_API_KEY"],
                                    base_url=config["OPENAI_BASE_URL"])
        self.tester_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous Senior QA Automation Engineer. 
            Your goal is to verify that the Coder's Python code perfectly fulfills the Architect's requirements.

            OPERATING RULES:
            1. EXECUTE: You MUST use 'python_repl_tool'. 
            IMPORTANT: Always use print() to display results. Bare expressions return nothing.
            Example: print(f"Result: add(3, 5)") and print(f"Assert: add(3,5) == 8")
            2. VERIFY: Use print(assert ...) style checks so you can SEE the output.
            3. FEEDBACK: 
            - If the code runs without error and the logic is 100% correct, reply ONLY with the word: Passed
            - If there is a SyntaxError, logic bug, or the result is incorrect, provide a concise but technical bug report. 
            
            STRICT FORMAT: 
            - If successful: "Passed"
            - If failure: "BUG REPORT: [Describe error and provide the traceback or output received from the REPL]"
            """),
            ("human", "Architect's Plan: {architect_out}\n\nCoder's Implementation:\n{coder_output}"),
            MessagesPlaceholder(variable_name="tester_feedback")
        ])
        self.tester = (
            {
                "coder_output": lambda x: x["coder_output"],
                "architect_out": lambda x: x["architect_out"],
                "tester_feedback": lambda x: x.get("tester_feedback", []),
            }
            | self.tester_prompt
            | self.tester_llm.bind_tools(tools=self.tools, tool_choice="auto")
        )
        self.agent = self._compose_graph(show_image=True)


    def _decomposition_node(self, state: SquadState, config: RunnableConfig) -> SquadState:
        # logger.info(f"[_decomposition_node] state['user_input']: {state['user_input']}")
        params = config.get("configurable", {})
        verbose = params.get("verbose", False)

        response = self.architect.invoke({"input": state["user_input"]})
        state["messages"].append(state["user_input"])
        state["architect_out"] = response.content
        if verbose:
            logger.info(f"[USER]: {state['user_input']}")
            logger.info(f"[AI_ARCHITECT]: {response.content}")
        return state
    
    def _coding_node(self, state: SquadState, config: RunnableConfig) -> dict:
        params = config.get("configurable", {})
        verbose = params.get("verbose", False)
        logger.info(f"[_coding_node] state['tester_feedback']: {state["tester_feedback"]}")
        logger.info(f"[_coding_node] state['architect_out']: {state["architect_out"]}")

        response = self.coder.invoke({"architect_out": state["architect_out"], "tester_feedback": state["tester_feedback"]})
        state["coder_output"] = response.content
        if verbose:
            logger.info(f"[AI_CODER]: {response.content}")
        
        return {
            "coder_output": response.content,
            "recode_attempts": state.get("recode_attempts", 0) + 1
        }
    
    def _testing_node(self, state: SquadState, config: RunnableConfig) -> dict:
        params = config.get("configurable", {})
        verbose = params.get("verbose", False)

        recent_feedback = list(state["tester_feedback"])[-4:]

        response = self.tester.invoke({
            "architect_out": state["architect_out"],
            "coder_output": state["coder_output"],
            "tester_feedback": recent_feedback
        })
        if verbose:
            logger.info(f"[AI_TESTER]: {response.content}")

        # add_messages reducer will accumulate messages
        return {
            "tester_feedback": [response],
            "recode_attempts": state.get("recode_attempts", 0)
        }

    
    def _if_errors_path(self, state: SquadState):
        last_message = state["tester_feedback"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue_testing"

        if isinstance(last_message, AIMessage) and "passed" in last_message.content.lower():
            return "complete"

        attempts = state.get("recode_attempts", 0)
        if attempts >= 3:
            logger.warning("Max recode attempts reached.")
            return "complete"

        return "recode"

    def _compose_graph(self, show_image: bool = True):
        graph = StateGraph(SquadState)

        graph.add_node("decomposition", self._decomposition_node)
        graph.add_node("coding", self._coding_node)
        graph.add_node("testing", self._testing_node)
        graph.add_node("tools", ToolNode(tools = self.tools, messages_key="tester_feedback")) # need to explicitly tell ToolNode to look for tool_calls in tester_feedback instead of messages by default

        graph.set_entry_point("decomposition")
        
        graph.add_edge("decomposition", "coding")
        graph.add_edge("coding", "testing")

        graph.add_conditional_edges(
            source="testing",
            path=self._if_errors_path,
            path_map={
                "continue_testing": "tools",
                "recode": "coding",
                "complete": END
            }
        )
        graph.add_edge("tools", "testing")
        agent = graph.compile()

        if show_image:
            with open("media/graph.png", "wb") as f:
                f.write(agent.get_graph().draw_mermaid_png())
            print("Graph saved to graph.png")   

        return agent

    def invoke(self, input_data: dict, verbose: bool = False) -> dict:
        init_state = SquadState(user_input=input_data["input"],
                                messages=[],
                                coder_output=None,
                                architect_out=None,
                                tester_feedback=[]
                                )
        response = self.agent.invoke(init_state, config={"verbose": verbose, "recursion_limit": 20})["coder_output"] # adding recursion limit for safety.
        return response

    async def event_generator(self, content, session_id):
        async for event in self.agent.astream(
            {"user_input": content, "messages": [], "tester_feedback": []},
            config={"recursion_limit": 20}
        ):
            for node_name, output in event.items():
                data = {
                    "node": node_name,
                    "content": output.get("coder_output") or output.get("architect_out") or "Processing...",
                    "session_id": session_id
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            await asyncio.sleep(0.1)
        