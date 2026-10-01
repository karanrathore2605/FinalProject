import ast
import math
import operator as op
import re
from typing import Union

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

MAX_ALLOWED_VALUE = 10**12


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates an AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if not math.isfinite(node.value):
                raise ValueError("Values must be finite numbers.")
            return node.value
        raise ValueError("Only numeric values are allowed in calculations.")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator '{operator_type.__name__}' is not allowed.")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        if operator_type == ast.Pow:
            if abs(right) > 100:
                raise ValueError("Exponent is too large (maximum allowed exponent is 100).")

        if operator_type == ast.Div and right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")

        result = ALLOWED_OPERATORS[operator_type](left, right)

        if not math.isfinite(result):
            raise ValueError("Calculation resulted in a non-finite number.")
        if abs(result) > MAX_ALLOWED_VALUE:
            raise ValueError("Calculation result exceeds maximum allowed magnitude.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError("Unary operator is not allowed.")

        operand = _safe_eval(node.operand)
        return ALLOWED_UNARY_OPERATORS[operator_type](operand)

    raise ValueError("Only pure mathematical expressions are allowed.")


def preprocess_expression(expr: str) -> str:
    """Preprocess human-typed mathematical expressions."""
    cleaned = expr.strip()
    # Replace unicode multiplication and division symbols
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    # Replace percentage signs like 15% with (15/100) or 0.15
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", cleaned)
    # Remove unwanted leading/trailing equality signs or question marks
    cleaned = cleaned.rstrip("=?").strip()
    return cleaned


def calculate(expression: str) -> Union[int, float]:
    """
    Safely evaluate a mathematical expression.
    Returns int if integer, else float.
    """
    cleaned = preprocess_expression(expression)
    if not cleaned:
        raise ValueError("Expression is empty.")

    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree)

        # Normalize floats that represent exact integers
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return round(result, 6)

    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")
    except SyntaxError:
        raise ValueError(f"Invalid mathematical expression: '{expression}'.")
    except Exception as exc:
        raise ValueError(str(exc))


def calculator_tool(expression: str) -> str:
    """
    Tool interface for calculator execution.
    Returns the calculation result as a string.
    """
    result = calculate(expression)
    return str(result)
