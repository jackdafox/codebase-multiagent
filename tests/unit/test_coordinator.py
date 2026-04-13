"""Unit tests for agents/coordinator.py routing logic."""

from __future__ import annotations

import os
from unittest.mock import patch

from codebase_rag.agents.coordinator import _route_validator, build_graph


class TestRouteValidator:
    """Test the validator routing logic."""

    def test_valid_result_routes_to_end(self):
        state = {
            "validation": {"is_valid": True, "errors": [], "warnings": [], "suggestions": []},
            "iteration": 1,
            "max_iterations": 3,
        }
        assert _route_validator(state) == "valid"

    def test_invalid_with_retries_routes_to_retry(self):
        state = {
            "validation": {"is_valid": False, "errors": ["type error"], "warnings": [], "suggestions": []},
            "iteration": 1,
            "max_iterations": 3,
        }
        assert _route_validator(state) == "retry"

    def test_invalid_at_max_iterations_routes_to_end(self):
        state = {
            "validation": {"is_valid": False, "errors": ["still broken"], "warnings": [], "suggestions": []},
            "iteration": 3,
            "max_iterations": 3,
        }
        assert _route_validator(state) == "max_iterations"

    def test_invalid_below_max_iterations_routes_to_retry(self):
        state = {
            "validation": {"is_valid": False, "errors": [], "warnings": [], "suggestions": []},
            "iteration": 2,
            "max_iterations": 3,
        }
        assert _route_validator(state) == "retry"

    def test_missing_validation_defaults_to_retry(self):
        state = {"iteration": 1, "max_iterations": 3}
        assert _route_validator(state) == "retry"

    def test_empty_validation_dict_defaults_to_retry(self):
        state = {"validation": {}, "iteration": 1, "max_iterations": 3}
        assert _route_validator(state) == "retry"

    def test_iteration_zero_routes_to_retry(self):
        state = {
            "validation": {"is_valid": False, "errors": [], "warnings": [], "suggestions": []},
            "iteration": 0,
            "max_iterations": 3,
        }
        assert _route_validator(state) == "retry"

    def test_env_default_max_iterations(self):
        with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "5"}, clear=False):
            state = {
                "validation": {"is_valid": False, "errors": [], "warnings": [], "suggestions": []},
                "iteration": 4,
                "max_iterations": 5,
            }
            assert _route_validator(state) == "retry"
            state["iteration"] = 5
            assert _route_validator(state) == "max_iterations"


class TestBuildGraph:
    """Test the LangGraph construction."""

    def test_graph_builds_successfully(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        graph = build_graph()
        nodes = list(graph.nodes)
        assert "architect" in nodes
        assert "engineer" in nodes
        assert "validator" in nodes

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None
