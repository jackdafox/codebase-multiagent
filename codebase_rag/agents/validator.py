"""Validator agent -- validates generated code using LangChain."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a senior code reviewer. Your job is to rigorously validate generated code.

Check for:
1. **Correctness** -- logic errors, off-by-ones, wrong APIs
2. **Safety** -- unhandled exceptions, SQL injection, hardcoded secrets, unsafe deserialization
3. **Completeness** -- all functions implemented, not stubbed
4. **Type correctness** -- type hints match actual behavior
5. **Style** -- matches language idioms, naming conventions
6. **Dependencies** -- imports exist, no circular deps

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
  "is_valid": true or false,
  "errors": ["list of critical errors that must be fixed"],
  "warnings": ["list of non-critical issues"],
  "suggestions": ["list of improvement suggestions"]
}

If the code is acceptable, set is_valid: true with empty errors."""


def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Validator node -- evaluates the Engineer's code.

    Args:
        state: Current graph state (must have code, language).

    Returns:
        Dict with validation field updated.
    """
    model_name = os.environ.get("VALIDATOR_MODEL", "gpt-4o")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    timeout = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60"))

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=0.0,
    )

    code = state.get("code", "")
    language = state.get("language", "python")

    user_message = f"""## Code to Validate
Target language: {language}

---
{code}
---

Respond with your validation as JSON:"""

    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
        )
        validation = _parse_validation(response.content)
    except Exception as exc:  # noqa: BLE001
        validation = {
            "is_valid": False,
            "errors": [f"Validator error: {exc}"],
            "warnings": [],
            "suggestions": [],
        }

    return {"validation": validation}


def _parse_validation(text: str) -> dict[str, Any]:
    """Parse the JSON response from the Validator LLM."""
    # Try to extract JSON from the response
    json_match = re.search(r"\{[\s\S]*\}", text.strip())
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "is_valid": False,
        "errors": [f"Could not parse validator response: {text[:200]}"],
        "warnings": [],
        "suggestions": [],
    }
