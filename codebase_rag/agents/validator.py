"""Validator agent -- validates generated code, returns errors or approval."""

from __future__ import annotations

import logging

from codebase_rag.agents.factory import get_client
from codebase_rag.agents.state import AgentState, ValidationResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior code reviewer. Your job is to rigorously validate generated code.

Check for:
1. **Correctness** -- logic errors, off-by-ones, wrong APIs
2. **Safety** -- unhandled exceptions, SQL injection, hardcoded secrets, unsafe deserialization
3. **Completeness** -- all functions implemented, not stubbed
4. **Type correctness** -- type hints match actual behavior
5. **Style** -- matches language idioms, naming conventions
6. **Dependencies** -- imports exist, no circular deps

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{{
  "is_valid": true or false,
  "errors": ["list of critical errors that must be fixed"],
  "warnings": ["list of non-critical issues"],
  "suggestions": ["list of improvement suggestions"]
}}

If the code is acceptable, set is_valid: true with empty errors."""


def run(state: AgentState) -> AgentState:
    """
    Run the Validator agent.

    Evaluates the Engineer's code and updates state.validation.

    Args:
        state: Current agent state (must have code set).

    Returns:
        Updated state with validation result filled in.
    """
    client = get_client("validator")
    model = getattr(client, "_model", "?") or "?"
    logger.info("Validator (model=%s) validating code (%d chars)", model, len(state.code))

    user_prompt = f"""## Code to Validate
Target language: {state.language}

---
{state.code}
---

Respond with your validation as JSON:"""

    try:
        response = client.complete(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
        )
        state.validation = _parse_validation(response)
        logger.info(
            "Validator result: is_valid=%s, errors=%d, warnings=%d",
            state.validation.is_valid,
            len(state.validation.errors),
            len(state.validation.warnings),
        )
    except Exception as exc:
        logger.error("Validator failed: %s -- treating as invalid", exc)
        state.validation = ValidationResult(
            is_valid=False,
            errors=[f"Validator error: {exc}"],
        )

    return state


def _parse_validation(response: str) -> ValidationResult:
    """Parse the JSON response from the Validator LLM."""
    import json
    import re

    # Try to extract JSON from the response (LLM might wrap in ```json ```)
    text = response.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return ValidationResult(
                is_valid=bool(data.get("is_valid", False)),
                errors=data.get("errors", []),
                warnings=data.get("warnings", []),
                suggestions=data.get("suggestions", []),
            )
        except json.JSONDecodeError:
            pass

    # Fallback: treat as invalid if we can't parse
    return ValidationResult(
        is_valid=False,
        errors=[f"Could not parse validator response: {text[:200]}"],
    )
