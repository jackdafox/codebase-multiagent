"""MultiAgentCoordinator -- orchestrates the Architect -> Engineer -> Validator loop."""

from __future__ import annotations

import logging
import os

from codebase_rag.agents import architect, engineer, validator
from codebase_rag.agents.state import AgentState, PipelineStatus
from codebase_rag.retrieval.rag import CodebaseRAG

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "3"))


class MultiAgentCoordinator:
    """
    Orchestrates the multi-agent code generation pipeline.

    Pipeline:
        1. Retrieve relevant context from RAG
        2. Architect: analyze query -> structured plan
        3. Engineer: generate code from plan
        4. Validator: validate code
        5. Loop back to Engineer with feedback until VALID or MAX_ITERATIONS

    Attributes:
        persist_directory: ChromaDB directory for RAG retrieval.
        max_iterations: Max Engineer<->Validator loops (default: 3).
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        max_iterations: int | None = None,
    ) -> None:
        self._persist_directory = persist_directory
        self._max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        self._rag: CodebaseRAG | None = None

    @property
    def rag(self) -> CodebaseRAG:
        """Lazy-load RAG client."""
        if self._rag is None:
            self._rag = CodebaseRAG(persist_directory=self._persist_directory)
        return self._rag

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
        max_iter = self._max_iterations

        # Initialize state
        state = AgentState(
            query=query,
            language=language,
            max_iterations=max_iter,
        )

        # Step 1: retrieve context from RAG
        logger.info("Retrieving RAG context for query: %s", query[:80])
        try:
            context = self.rag.get_context(
                question=query,
                language=language,
                n_results=n_results,
            )
            state.context = context
        except Exception as exc:
            logger.warning("RAG retrieval failed (continuing without context): %s", exc)
            state.context = ""

        # Step 2: Architect generates plan
        state = architect.run(state)
        if state.status == PipelineStatus.FAILED:
            return state

        # Step 3-5: Engineer -> Validator loop
        while True:
            state.iteration += 1
            logger.info("=== Iteration %d/%d ===", state.iteration, max_iter)

            # Engineer generates code
            state = engineer.run(state)
            if state.status == PipelineStatus.FAILED:
                return state

            # Validator checks code
            state = validator.run(state)
            if state.status == PipelineStatus.FAILED:
                return state

            if state.validation.is_valid:
                state.status = PipelineStatus.VALIDATED
                logger.info("Code validated on iteration %d", state.iteration)
                return state

            if not state.can_iterate():
                state.status = PipelineStatus.MAX_ITERATIONS_REACHED
                logger.warning(
                    "Max iterations (%d) reached -- code not validated. "
                    "Errors: %s", max_iter, state.validation.errors
                )
                return state

            logger.info(
                "Validation failed (iter %d) -- %d errors, looping back to Engineer",
                state.iteration,
                len(state.validation.errors),
            )
