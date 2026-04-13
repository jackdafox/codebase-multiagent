"""Engineer agent -- generates code based on the Architect's plan using LangChain."""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a senior software engineer. Your job is to write high-quality code based on the architect's plan.

Generate clean, working code that:
- Matches the target language conventions
- Follows best practices (error handling, type hints, etc.)
- Is ready to use (not pseudocode)
- Includes docstrings for public functions

Return ONLY the code -- no markdown code fences unless specifically requested."""


def engineer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Engineer node -- generates code from the Architect's plan.

    Args:
        state: Current graph state (must have query, language, plan, context).

    Returns:
        Dict with code field updated, and iteration incremented.
    """
    model_name = os.environ.get("ENGINEER_MODEL", "gpt-4o")
    api_key = os.environ.get("OPENAI_API_KEY") or None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    timeout = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60"))

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,  # type: ignore[arg-type]
        base_url=base_url,
        timeout=timeout,
        temperature=0.2,
    )

    query = state["query"]
    language = state.get("language", "python")
    plan = state.get("plan", "(no plan -- use general knowledge)")
    context = state.get("context", "") or "(no context retrieved)"
    iteration = state.get("iteration", 0)

    # Build feedback from previous validation errors (if any)
    validation = state.get("validation", {})
    prev_errors = validation.get("errors", []) if isinstance(validation, dict) else []
    feedback = "\n".join(prev_errors) if prev_errors and iteration > 0 else "none"

    user_message = f"""## User Request
{query}

## Target Language
{language}

## Architect's Plan
{plan}

## Retrieved Code Context
---
{context}
---

## Previous Validation Feedback
(if any -- address these issues in your code)
---
{feedback}
---

Generate the code now:"""

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    code = cast(str, response.content).strip()

    return {
        "code": code,
        "iteration": iteration + 1,
    }
