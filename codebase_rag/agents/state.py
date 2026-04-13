"""Shared state passed between agents in the multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(Enum):
    """Roles in the multi-agent pipeline."""

    ARCHITECT = "architect"
    ENGINEER = "engineer"
    VALIDATOR = "validator"


class PipelineStatus(Enum):
    """Overall pipeline status."""

    RUNNING = "running"
    VALIDATED = "validated"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result from the Validator agent."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    """
    Shared state carried through the Architect -> Engineer -> Validator loop.

    Attributes:
        query: The original user request.
        language: Target language (e.g., "python").
        context: Retrieved code chunks from RAG (joined string).
        plan: Architect's structured plan.
        code: Engineer-generated code.
        validation: Validator's result.
        iteration: Current loop iteration (0-indexed).
        max_iterations: Max Engineer<->Validator loops.
        status: Overall pipeline status.
        error: Error message if pipeline failed.
        metadata: Additional context (file paths, etc.).
    """

    query: str
    language: str = "python"
    context: str = ""
    plan: str = ""
    code: str = ""
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(is_valid=False)
    )
    iteration: int = 0
    max_iterations: int = 3
    status: PipelineStatus = PipelineStatus.RUNNING
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_iterate(self) -> bool:
        """Check if another Engineer<->Validator loop is allowed."""
        return (
            self.status == PipelineStatus.RUNNING
            and self.iteration < self.max_iterations
            and not self.validation.is_valid
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for logging/debugging."""
        return {
            "query": self.query,
            "language": self.language,
            "plan": self.plan,
            "code": self.code[:200] + "..." if len(self.code) > 200 else self.code,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "iteration": self.iteration,
            "status": self.status.value,
        }
