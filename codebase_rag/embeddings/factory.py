"""OpenAI embedding factory compatible with Langchain."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import openai

DEFAULT_MODEL = "text-embedding-3-small"
_EMBEDDING_LOCK = threading.Lock()


class OpenAIEmbeddingsFactory:
    """
    Factory for creating OpenAI embedding functions.

    Produces embeddings compatible with Langchain's vectorstore interface.
    Thread-safe singleton per model name.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize the embedding factory.

        Args:
            api_key: OpenAI API key. If None, read from OPENAI_API_KEY env var.
            model: Embedding model name.
            dimensions: Embedding dimensions (for text-embedding-3).
            batch_size: Max chunks per batch for embedding requests.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key must be provided or set as OPENAI_API_KEY env var"
            )
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client: "openai.OpenAI | None" = None

    @property
    def client(self) -> "openai.OpenAI":
        """Lazy-load the OpenAI client thread-safely."""
        if self._client is None:
            with _EMBEDDING_LOCK:
                if self._client is None:
                    import openai
                    self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query string.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self._model,
            input=[text],
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """
        Langchain-compatible callable interface.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embed_texts(texts)


# Global factory instance per model
_FACTORIES: dict[str, OpenAIEmbeddingsFactory] = {}


def get_embedding_function(
    model: str = DEFAULT_MODEL,
    dimensions: int | None = None,
    api_key: str | None = None,
) -> OpenAIEmbeddingsFactory:
    """
    Get or create a singleton embedding factory for the given model.

    Args:
        model: Embedding model name.
        dimensions: Embedding dimensions.
        api_key: OpenAI API key.

    Returns:
        An OpenAIEmbeddingsFactory instance.
    """
    key = f"{model}:{dimensions}:{api_key}"
    if key not in _FACTORIES:
        with _EMBEDDING_LOCK:
            if key not in _FACTORIES:
                _FACTORIES[key] = OpenAIEmbeddingsFactory(
                    api_key=api_key,
                    model=model,
                    dimensions=dimensions,
                )
    return _FACTORIES[key]
