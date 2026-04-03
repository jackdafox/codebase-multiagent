"""Codebase RAG — Retrieval Augmented Generation for codebases."""

from codebase_rag.core.chunk import CodeChunk
from codebase_rag.indexing.indexer import CodebaseIndexer
from codebase_rag.retrieval.rag import CodebaseRAG

__version__ = "0.1.0"

__all__ = [
    "CodeChunk",
    "CodebaseIndexer",
    "CodebaseRAG",
]
