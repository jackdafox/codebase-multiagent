"""Engineer agent -- generates code based on the Architect's plan."""

from __future__ import annotations

import logging

from codebase_rag.agents.factory import get_client
from codebase_rag.agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer. Your job is to write high-quality code based on the architect's plan.

Generate clean, working code that:
- Matches the target language conventions
- Follows best practices (error handling, type hints, etc.)
- Is ready to use (not pseudocode)
- Includes docstrings for public functions

Return ONLY the code -- no markdown code fences unless specifically requested."""

USER_TEMPLATE = """## User Request
{query}

## Target Language
{language}

## Architect's Plan
{plan}

## Retrieved Code Context
(relevant existing code from the codebase)
---
{context}
---

## Previous Validation Feedback
(if any -- address these issues in your code)
---
{feedback}
---

Generate the code now:"""


def run(state: AgentState) -> AgentState:
    """
    Run the Engineer agent.

    Generates code from the Architect's plan and RAG context.
    If validation feedback exists from a previous iteration, includes it.

    Args:
        state: Current agent state (must have plan, context set).

    Returns:
        Updated state with code filled in.
    """
    client = get_client("engineer")
    model = getattr(client, "_model", "?") or "?"
    logger.info(
        "Engineer (model=%s, iter=%d) generating code",
        model, state.iteration,
    )

    feedback = ""
    if state.validation.errors and state.iteration > 0:
        feedback = "\n".join(state.validation.errors)

    user_prompt = USER_TEMPLATE.format(
        query=state.query,
        language=state.language,
        plan=state.plan or "(no plan -- use general knowledge)",
        context=state.context or "(no context retrieved)",
        feedback=feedback or "none",
    )

    try:
        code = client.complete(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )
        state.code = code.strip()
        logger.info("Engineer produced code (%d chars)", len(state.code))
    except Exception as exc:
        logger.error("Engineer failed: %s", exc)
        state.error = f"Engineer error: {exc}"
        state.status = state.status.FAILED

    return state
