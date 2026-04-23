"""
LLM with tool calling demo.

Sends a user prompt to an OpenAI model, lets the model decide whether to
invoke a local calculator tool, executes the tool when requested, feeds the
result back to the model, and prints the model's final answer.
"""

from __future__ import annotations

import ast
import json
import operator
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


_ALLOWED_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed expression tree with a whitelist of ops."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        return _ALLOWED_BIN_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> float:
    """Evaluate a math expression safely.

    Only numeric literals and the operators + - * / // % ** are permitted.
    Names, function calls, attribute access and anything else are rejected.
    """
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression and return the numeric "
                "result. Supports + - * / // % ** and parentheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '(2 + 3) * 4'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    }
]


def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "calculator":
        try:
            result = calculator(arguments["expression"])
            return json.dumps({"result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return json.dumps({"error": f"Unknown tool: {name}"})


def run(prompt: str) -> str:
    client = OpenAI()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. When the user asks a question "
                "that requires arithmetic, call the `calculator` tool instead "
                "of computing the result yourself."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    choice = first.choices[0].message

    if not choice.tool_calls:
        return choice.content or ""

    messages.append(
        {
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
        }
    )

    for tool_call in choice.tool_calls:
        args = json.loads(tool_call.function.arguments or "{}")
        output = _dispatch_tool(tool_call.function.name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            }
        )

    second = client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content or ""


def main() -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Copy .env.example to .env.", file=sys.stderr)
        return 1

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        try:
            prompt = input("Ask something: ").strip()
        except EOFError:
            prompt = ""

    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        return 1

    answer = run(prompt)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
