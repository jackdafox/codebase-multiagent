"""Embeddings module exports."""

from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory, get_embedding_function

__all__ = [
    "OpenAIEmbeddingsFactory",
    "get_embedding_function",
]
