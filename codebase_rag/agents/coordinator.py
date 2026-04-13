"""MultiAgentCoordinator -- LangGraph-based orchestration of the A->E->V pipeline."""

from __future__ import annotations

import logging
import os

from langgraph.graph import END, START, StateGraph

from codebase_rag.agents import architect, engineer, validator
from codebase_rag.agents.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "3"))


def _route_validator(state: AgentState) -> str:
    """Route based on validation result."""
    validation = state.get("validation", {})
    is_valid = validation.get("is_valid", False) if isinstance(validation, dict) else False
    if is_valid:
        return "valid"
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if iteration >= max_iter:
        return "max_iterations"
    return "retry"


def build_graph() -> StateGraph:
    """Build the LangGraph state graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("architect", architect.architect_node)
    graph.add_node("engineer", engineer.engineer_node)
    graph.add_node("validator", validator.validator_node)

    # Edges
    graph.add_edge(START, "architect")
    graph.add_edge("architect", "engineer")
    graph.add_edge("engineer", "validator")

    # Conditional routing from validator
    graph.add_conditional_edges(
        "validator",
        _route_validator,
        {
            "valid": END,
            "retry": "engineer",
            "max_iterations": END,
        },
    )

    return graph


def run(
    query: str,
    language: str = "python",
    context: str = "",
    max_iterations: int | None = None,
) -> AgentState:
    """
    Run the multi-agent pipeline via LangGraph.

    Args:
        query: The user's code request.
        language: Target programming language.
        context: Pre-retrieved RAG context (optional).
        max_iterations: Max Engineer<->Validator loops.

    Returns:
        Final AgentState with code and validation result.
    """
    max_iter = max_iterations or DEFAULT_MAX_ITERATIONS

    initial_state: AgentState = {
        "query": query,
        "language": language,
        "context": context or "",
        "plan": "",
        "code": "",
        "validation": {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        },
        "iteration": 0,
        "max_iterations": max_iter,
        "status": "running",
        "error": None,
    }

    compiled = build_graph().compile()
    result: AgentState = compiled.invoke(initial_state)  # type: ignore[arg-type]
    return result


class MultiAgentCoordinator:
    """
    Orchestrates the multi-agent code generation pipeline via LangGraph.

    Pipeline:
        1. Retrieve relevant context from RAG
        2. Architect: analyze query -> structured plan
        3. Engineer: generate code from plan
        4. Validator: validate code
        5. Loop back to Engineer with feedback until VALID or MAX_ITERATIONS
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        max_iterations: int | None = None,
    ) -> None:
        self._persist_directory = persist_directory
        self._max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS

    def run(
        self,
        query: str,
        language: str = "python",
        n_results: int = 5,
    ) -> AgentState:
        """
        Run the full multi-agent pipeline.

        Args:
            query: The user's code request.
            language: Target programming language.
            n_results: Number of RAG context chunks to retrieve.

        Returns:
            Final AgentState with code and validation result.
        """
        import logging
        from codebase_rag.retrieval.rag import CodebaseRAG

        logger = logging.getLogger(__name__)
        max_iter = self._max_iterations

        # Retrieve context from RAG
        context = ""
        try:
            rag = CodebaseRAG(persist_directory=self._persist_directory)
            context = rag.get_context(
                question=query,
                language=language,
                n_results=n_results,
            )
        except Exception as exc:
            logger.warning("RAG retrieval failed (continuing without context): %s", exc)

        return run(
            query=query,
            language=language,
            context=context,
            max_iterations=max_iter,
        )