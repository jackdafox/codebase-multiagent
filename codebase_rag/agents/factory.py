"""LLM client factory -- reads from env vars, supports OpenAI / Anthropic / Ollama."""

from __future__ import annotations

import os
from typing import Any

import requests

# Default env var names -- override per-agent if needed
PROVIDER = os.environ.get("AGENT_PROVIDER", "openai")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TIMEOUT = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60"))


class LLMClient:
    """Unified LLM client wrapping OpenAI / Anthropic / Ollama."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._provider = (provider or PROVIDER).lower()
        self._model = model
        self._api_key = api_key or self._get_key()
        self._base_url = base_url or self._get_base_url()
        self._timeout = timeout or TIMEOUT

    def _get_key(self) -> str:
        if self._provider == "anthropic":
            return ANTHROPIC_KEY
        return OPENAI_KEY

    def _get_base_url(self) -> str:
        if self._provider == "ollama":
            return OLLAMA_BASE_URL
        if self._provider == "openai":
            return OPENAI_BASE_URL
        return OPENAI_BASE_URL

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a completion request to the configured LLM.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            model: Override model for this call.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            The LLM's text response.
        """
        model = model or self._model
        if not model:
            raise ValueError("No model specified")

        if self._provider == "anthropic":
            return self._anthropic_complete(prompt, system, model, max_tokens)
        return self._openai_complete(prompt, system, model, temperature, max_tokens)

    def _openai_complete(
        self,
        prompt: str,
        system: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _anthropic_complete(
        self,
        prompt: str,
        system: str | None,
        model: str,
        max_tokens: int,
    ) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        messages = [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


def get_client(
    role: str,
    model: str | None = None,
    provider: str | None = None,
) -> LLMClient:
    """
    Factory to get an LLM client for a given agent role.

    Reads role-specific env vars if model/provider not provided:
      {ROLE}_MODEL  (e.g., ENGINEER_MODEL)
      {ROLE}_PROVIDER (e.g., ENGINEER_PROVIDER)

    Args:
        role: Agent role (architect, engineer, validator).
        model: Override model.
        provider: Override provider.

    Returns:
        Configured LLMClient instance.
    """
    role_upper = role.upper()
    model_env = os.environ.get(f"{role_upper}_MODEL")
    provider_env = os.environ.get(f"{role_upper}_PROVIDER")

    return LLMClient(
        provider=provider or provider_env,
        model=model or model_env,
    )
