"""Agent state using LangGraph's MessagesState."""

from __future__ import annotations

from typing import TypedDict


class ValidationResult(TypedDict):
    """Validation result from the Validator agent."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]


class AgentState(TypedDict):
    """
    State carried through the multi-agent graph.

    Extends MessagesState with additional fields.
    """

    query: str
    language: str
    context: str
    plan: str
    code: str
    validation: ValidationResult
    iteration: int
    max_iterations: int
    status: str  # running | validated | max_iterations_reached | failed
    error: str | None
