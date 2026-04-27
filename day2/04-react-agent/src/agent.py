"""Helper utilities for ReAct Agent notebook."""

from typing import Any


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer."""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract b from a and returns the result integer."""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide a by b and returns the result float."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def format_agent_response(response: Any) -> str:
    """Format agent response for display."""
    sources_count = len(response.sources) if hasattr(response, "sources") else 0

    return f"""
## Response
{response.response}

## Sources Used
{sources_count} source(s)
"""


def extract_thought_action(agent_output: str) -> dict:
    """Extract Thought and Action from agent verbose output.

    Args:
        agent_output: Raw output from ReAct agent

    Returns:
        dict with 'thought' and 'action' keys
    """
    lines = agent_output.split("\n")
    thought = None
    action = None

    for line in lines:
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip()

    return {"thought": thought, "action": action}
