# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Safe evaluation of numeric expressions."""

from __future__ import annotations

import ast
import math
from typing import Callable

_RATIO_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_RATIO_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval_numeric_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Boolean literals are not allowed")
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"Unsupported literal: {value!r}")

    if isinstance(node, ast.BinOp):
        lhs = _eval_numeric_expr(node.left)
        rhs = _eval_numeric_expr(node.right)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            return lhs / rhs
        if isinstance(node.op, ast.Pow):
            return lhs**rhs
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        value = _eval_numeric_expr(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Name):
        if node.id in _RATIO_CONSTANTS:
            return _RATIO_CONSTANTS[node.id]
        raise ValueError(f"Unknown symbol: {node.id!r}")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple math function names are supported")
        fn_name = node.func.id
        fn = _RATIO_FUNCTIONS.get(fn_name)
        if fn is None:
            raise ValueError(f"Unsupported function: {fn_name!r}")
        args = [_eval_numeric_expr(arg) for arg in node.args]
        return float(fn(*args))

    raise ValueError(f"Unsupported expression token: {type(node).__name__}")


def parse_numeric_expression(value: str, *, context: str = "value") -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} cannot be empty")

    normalized = text.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{context} is not a valid numeric expression: {value!r}") from exc

    try:
        out = float(_eval_numeric_expr(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError(f"{context} contains division by zero: {value!r}") from exc
    except OverflowError as exc:
        raise ValueError(f"{context} overflowed while evaluating: {value!r}") from exc
    except TypeError as exc:
        raise ValueError(f"{context} is invalid: {value!r} ({exc})") from exc
    except ValueError as exc:
        raise ValueError(f"{context} is invalid: {value!r} ({exc})") from exc

    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite: {value!r}")
    return out
