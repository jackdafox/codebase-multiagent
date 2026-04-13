"""Architect agent -- analyzes the query and plans the approach."""

from __future__ import annotations

import logging

from codebase_rag.agents.factory import get_client, LLMClient
from codebase_rag.agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software architect. Your job is to analyze a user's code request and produce a structured plan.

Given the user's query and relevant code context retrieved from their codebase, produce a clear, actionable plan describing:
1. What needs to be built or changed
2. Which files or modules are affected
3. Key functions/classes to implement or modify
4. Any dependencies or considerations

Be specific about the approach. Output a numbered plan in plain text."""

USER_TEMPLATE = """## User Request
{query}

## Language
{target_language}

## Retrieved Code Context
(context from their codebase -- relevant existing code)
---
{context}
---
"""


def run(state: AgentState) -> AgentState:
    """
    Run the Architect agent.

    Populates state.plan with a structured approach based on the query
    and retrieved RAG context.

    Args:
        state: Current agent state (must have query, language, context set).

    Returns:
        Updated state with plan filled in.
    """
    client = get_client("architect")
    model = _get_model_name(client)
    logger.info("Architect (model=%s) generating plan for: %s", model, state.query[:80])

    user_prompt = USER_TEMPLATE.format(
        query=state.query,
        target_language=state.language,
        context=state.context or "(no context retrieved -- use general knowledge)",
    )

    try:
        plan = client.complete(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
        )
        state.plan = plan.strip()
        logger.info("Architect produced plan (%d chars)", len(state.plan))
    except Exception as exc:
        logger.error("Architect failed: %s", exc)
        state.error = f"Architect error: {exc}"
        state.status = state.status.FAILED

    return state


def _get_model_name(client: "LLMClient") -> str:
    """Extract model name for logging."""
    return getattr(client, "_model", "?") or "?"
