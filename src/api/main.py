from src.api.agent.agent import AgentSquad
from src.logging import logger

if __name__ == "__main__":
    agent_squad = AgentSquad()

    user_input = "Calculate the 50th Fibonacci number using an iterative approach and print the result."
    response = agent_squad.invoke({"input": user_input})
    logger.info(response)
