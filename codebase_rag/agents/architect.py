"""Architect agent -- analyzes query and produces a structured plan using LangChain."""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a senior software architect. Your job is to analyze a user's code request and produce a structured plan.

Given the user's query and relevant code context retrieved from their codebase, produce a clear, actionable plan describing:
1. What needs to be built or changed
2. Which files or modules are affected
3. Key functions/classes to implement or modify
4. Any dependencies or considerations

Be specific about the approach. Output a numbered plan in plain text."""


def architect_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Architect node -- analyzes query and produces a structured plan.

    Args:
        state: Current graph state (must have query, language, context).

    Returns:
        Dict with plan field updated.
    """
    model_name = os.environ.get("ARCHITECT_MODEL", "gpt-4o")
    api_key = os.environ.get("OPENAI_API_KEY") or None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    timeout = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60"))

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,  # type: ignore[arg-type]
        base_url=base_url,
        timeout=timeout,
        temperature=0.0,
    )

    context = state.get("context", "") or "(no context retrieved -- use general knowledge)"
    query = state["query"]
    language = state.get("language", "python")

    user_message = f"""## User Request
{query}

## Language
{language}

## Retrieved Code Context
---
{context}
---

Produce your structured plan:"""

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    plan = cast(str, response.content).strip()

    return {"plan": plan}
