from typing import TypedDict 
import os
from dotenv import load_dotenv

load_dotenv()

class Config(TypedDict):
    OPENAI_API_KEY: str

config: Config = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "NO_KEY"),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "default"),
    "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", True),
    "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "default"),
    "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY", "NO_KEY"),
    "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
}