"""Tool definitions and schemas for hello-agent.

Math evaluation uses a restricted AST walk, not eval(). eval() with
{"__builtins__": None} still executes arbitrary node types under the
hood and is easy to get subtly wrong; an explicit allow-list of AST
node types is the safer, auditable choice (see vs-literature.md).
"""
import ast
import operator
from typing import Optional

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def get_weather(location: str) -> str:
    return f"The current weather in {location} is 22°C (72°F) and sunny."


def calculate_expression(expression: str) -> str:
    try:
        node = ast.parse(expression, mode="eval").body
        return f"Result: {_safe_eval(node)}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOL_MAP = {"get_weather": get_weather, "calculate_expression": calculate_expression}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g., 25 * 4 + 10"}},
                "required": ["expression"],
            },
        },
    },
]


def validate_arguments(tool_name: str, args: dict) -> Optional[str]:
    """Checks parsed arguments against the declared schema before invocation.
    Returns an error string, or None if valid. Catches what the original
    implementation silently trusted: missing required fields and wrong types.
    """
    if tool_name not in TOOL_MAP:
        return f"unknown tool '{tool_name}'"
    schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == tool_name)
    props = schema["parameters"]["properties"]
    required = schema["parameters"]["required"]

    for field in required:
        if field not in args:
            return f"missing required argument '{field}'"
    for field, value in args.items():
        if field not in props:
            return f"unexpected argument '{field}'"
        if props[field]["type"] == "string" and not isinstance(value, str):
            return f"argument '{field}' must be a string, got {type(value).__name__}"
    return None