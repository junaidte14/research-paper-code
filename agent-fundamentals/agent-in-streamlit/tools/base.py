# tools/base.py - Abstract Base Tool & Schema Decorator
import inspect
import functools
from typing import Callable, Any, Dict

class BaseTool:
    """
    Wraps a callable Python function and auto-generates 
    an OpenAPI / Groq compatible JSON tool schema.
    """
    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__.strip() if func.__doc__ else "No description provided.")
        self.schema = self._generate_groq_schema()

    def _generate_groq_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.func)
        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for param_name, param in sig.parameters.items():
            param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"

def tool(name: str = None, description: str = None):
    """Decorator to convert a standard function into a BaseTool"""
    def decorator(func):
        tool_obj = BaseTool(func, name=name, description=description)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.tool_obj = tool_obj
        return wrapper
    return decorator
