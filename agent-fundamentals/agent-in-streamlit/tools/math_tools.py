# tools/math_tools.py - Safe Abstract Syntax Tree Math Tool
import ast
import operator
from tools.base import tool

# Allowed math operators for secure evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
    raise ValueError("Unsafe or invalid math expression")

@tool(description="Evaluates mathematical expressions safely using Python AST without using eval()")
def calculate_math_expression(expression: str) -> str:
    try:
        node = ast.parse(expression, mode='eval').body
        result = _safe_eval(node)
        return f"Result: {result}"
    except Exception as e:
        return f"Math Evaluation Error: {str(e)}"
