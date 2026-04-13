"""Unit tests for agents/state.py."""

from __future__ import annotations

from codebase_rag.agents.state import AgentState, ValidationResult


class TestValidationResult:
    def test_valid_result(self):
        result: ValidationResult = {
            "is_valid": True,
            "errors": [],
            "warnings": ["unused import"],
            "suggestions": ["could use dataclass"],
        }
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_invalid_result(self):
        result: ValidationResult = {
            "is_valid": False,
            "errors": ["missing return type", "undefined name 'foo'"],
            "warnings": [],
            "suggestions": [],
        }
        assert result["is_valid"] is False
        assert len(result["errors"]) == 2

    def test_empty_lists(self):
        result: ValidationResult = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }
        assert result["errors"] == []
        assert result["warnings"] == []
        assert result["suggestions"] == []


class TestAgentState:
    def test_initial_state(self):
        state: AgentState = {
            "query": "add auth",
            "language": "python",
            "context": "...",
            "plan": "",
            "code": "",
            "validation": {
                "is_valid": False,
                "errors": [],
                "warnings": [],
                "suggestions": [],
            },
            "iteration": 0,
            "max_iterations": 3,
            "status": "running",
            "error": None,
        }
        assert state["query"] == "add auth"
        assert state["max_iterations"] == 3
        assert state["status"] == "running"
        assert state["error"] is None

    def test_iteration_tracking(self):
        state: AgentState = {
            "query": "add auth",
            "language": "python",
            "context": "...",
            "plan": "",
            "code": "def auth(): pass",
            "validation": {
                "is_valid": False,
                "errors": ["missing return"],
                "warnings": [],
                "suggestions": [],
            },
            "iteration": 1,
            "max_iterations": 3,
            "status": "running",
            "error": None,
        }
        assert state["iteration"] < state["max_iterations"]

    def test_validated_state(self):
        state: AgentState = {
            "query": "add auth",
            "language": "python",
            "context": "...",
            "plan": "step 1...",
            "code": "def auth(): return True",
            "validation": {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": [],
            },
            "iteration": 2,
            "max_iterations": 3,
            "status": "validated",
            "error": None,
        }
        assert state["status"] == "validated"
        assert state["validation"]["is_valid"] is True

    def test_max_iterations_state(self):
        state: AgentState = {
            "query": "add auth",
            "language": "python",
            "context": "...",
            "plan": "step 1...",
            "code": "def auth(): pass",
            "validation": {
                "is_valid": False,
                "errors": ["still broken"],
                "warnings": [],
                "suggestions": [],
            },
            "iteration": 3,
            "max_iterations": 3,
            "status": "max_iterations_reached",
            "error": None,
        }
        assert state["iteration"] == state["max_iterations"]
        assert state["status"] == "max_iterations_reached"
        assert state["validation"]["is_valid"] is False

    def test_failed_state(self):
        state: AgentState = {
            "query": "add auth",
            "language": "python",
            "context": "...",
            "plan": "",
            "code": "",
            "validation": {
                "is_valid": False,
                "errors": [],
                "warnings": [],
                "suggestions": [],
            },
            "iteration": 0,
            "max_iterations": 3,
            "status": "failed",
            "error": "OpenAI API rate limit exceeded",
        }
        assert state["status"] == "failed"
        assert state["error"] is not None
