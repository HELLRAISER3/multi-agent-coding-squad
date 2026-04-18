from langchain_experimental.utilities.python import PythonREPL
from langchain_core.tools import tool

repl = PythonREPL()

@tool
def python_repl_tool(code: str):
    """Executes python code and returns the standard output."""
    try:
        return repl.run(code)
    except Exception as e:
        return f"Error: {str(e)}"