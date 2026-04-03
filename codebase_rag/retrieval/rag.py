"""CodebaseRAG — querying and context retrieval from indexed codebases."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from codebase_rag.core.language import is_supported_language
from codebase_rag.db.client import get_client
from codebase_rag.db.collections import CollectionManager
from codebase_rag.embeddings.factory import (
    DEFAULT_MODEL,
    get_embedding_function,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """A single retrieved chunk."""

    id: str
    text: str
    language: str
    node_type: str
    file_path: str
    start_line: int
    end_line: int
    fully_qualified_name: str | None
    parent_name: str | None
    docstring: str | None
    signature: str | None
    score: float | None = None


@dataclass
class QueryResult:
    """Result of a RAG query."""

    chunks: list[ChunkResult] = field(default_factory=list)
    total_results: int = 0
    languages_queried: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.chunks)


class CodebaseRAG:
    """
    RAG system for querying indexed codebases.

    Wraps ChromaDB per-language collections, provides retrieval with
    metadata filtering, and formats context for LLM consumption.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = DEFAULT_MODEL,
        embedding_dimensions: int | None = None,
        embedding_api_key: str | None = None,
    ) -> None:
        """
        Initialize the RAG system.

        Args:
            persist_directory: ChromaDB persistence directory.
            embedding_model: OpenAI embedding model name.
            embedding_dimensions: Embedding dimensions.
            embedding_api_key: OpenAI API key override.
        """
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

        self._client = get_client(persist_directory)
        self._collections = CollectionManager(self._client)
        self._embedding_factory = get_embedding_function(
            model=embedding_model,
            dimensions=embedding_dimensions,
            api_key=embedding_api_key,
        )

    def query(
        self,
        question: str,
        language: str | None = None,
        file_paths: list[str] | None = None,
        n_results: int = 5,
        include_chunks: bool = True,
        include_scores: bool = True,
    ) -> QueryResult:
        """
        Query the indexed codebase.

        Args:
            question: Natural language question.
            language: Restrict search to a specific language.
            file_paths: Restrict search to specific files.
            n_results: Number of chunks to retrieve.
            include_chunks: Include full chunk text in results.
            include_scores: Include relevance scores.

        Returns:
            QueryResult with retrieved chunks and metadata.
        """
        query_embedding = self._embedding_factory.embed_query(question)

        languages = [language] if language else self._collections.get_all_languages()
        result = QueryResult(languages_queried=set(languages))

        # Build metadata filter
        where: dict[str, Any] | None = None
        if file_paths:
            if len(file_paths) == 1:
                where = {"file_path": file_paths[0]}
            else:
                where = {"file_path": {"$in": file_paths}}

        all_chunks: list[ChunkResult] = []

        for lang in languages:
            if not is_supported_language(lang):
                continue

            try:
                query_result = self._collections.query(
                    language=lang,
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where=where,
                )
            except Exception as e:
                logger.warning("Query failed for language %s: %s", lang, e)
                continue

            if not query_result.get("ids"):
                continue

            # Extract chunks from query result
            ids = query_result["ids"][0]
            distances = query_result.get("distances", [[]])[0]
            documents = query_result.get("documents", [[]])[0]
            metadatas = query_result.get("metadatas", [[]])[0]

            for i, chunk_id in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""
                score = distances[i] if i < len(distances) else None

                chunk = ChunkResult(
                    id=chunk_id,
                    text=doc if include_chunks else "",
                    language=meta.get("language", lang),
                    node_type=meta.get("node_type", "unknown"),
                    file_path=meta.get("file_path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    fully_qualified_name=meta.get("fully_qualified_name"),
                    parent_name=meta.get("parent_name"),
                    docstring=meta.get("docstring") or None,
                    signature=meta.get("signature") or None,
                    score=score if include_scores else None,
                )
                all_chunks.append(chunk)

        # Sort by score (distance) ascending
        all_chunks.sort(key=lambda c: c.score if c.score is not None else float("inf"))
        result.chunks = all_chunks
        result.total_results = len(all_chunks)

        return result

    def get_context(
        self,
        question: str,
        language: str | None = None,
        file_paths: list[str] | None = None,
        n_results: int = 5,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get a formatted context string suitable for an LLM prompt.

        Args:
            question: The user's question.
            language: Language to restrict search to.
            file_paths: File paths to restrict search to.
            n_results: Number of chunks to retrieve.
            max_tokens: Approximate max tokens for context string.

        Returns:
            Formatted context string with file paths, line numbers, and code.
        """
        query_result = self.query(
            question=question,
            language=language,
            file_paths=file_paths,
            n_results=n_results,
            include_chunks=True,
            include_scores=True,
        )

        if not query_result.chunks:
            return "No relevant code found."

        context_parts = []
        total_chars = 0

        for chunk in query_result.chunks:
            # Rough token estimate: 1 token ≈ 4 chars
            estimated_tokens = len(chunk.text) // 4
            if total_chars + estimated_tokens > max_tokens * 4:
                break

            header = f"// {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
            if chunk.fully_qualified_name:
                header += f" ({chunk.fully_qualified_name})"
            header += f" [{chunk.node_type}]"

            context_parts.append(f"{header}\n{chunk.text}")
            total_chars += len(chunk.text) + len(header)

        return "\n\n---\n\n".join(context_parts)

    def get_all_indexed_languages(self) -> list[str]:
        """Return list of languages that have indexed chunks."""
        return self._collections.get_all_languages()
