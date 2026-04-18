from typing import TypedDict 
import os
from dotenv import load_dotenv

load_dotenv()

class Config(TypedDict):
    OPENAI_API_KEY: str

config: Config = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "NO_KEY"),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "NO_KEY"),
}