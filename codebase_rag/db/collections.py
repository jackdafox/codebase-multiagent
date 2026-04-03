"""Per-language ChromaDB collection management."""

from __future__ import annotations

from typing import Any

from chromadb.errors import ChromaError

COLLECTION_NAME_PREFIX = "codebase"
COLLECTION_SUFFIX = "chunks"


def _collection_name(language: str) -> str:
    """Build the ChromaDB collection name for a language."""
    return f"{COLLECTION_NAME_PREFIX}_{language}_{COLLECTION_SUFFIX}"


class CollectionManager:
    """
    Manages per-language ChromaDB collections.

    Each language gets its own collection with a consistent schema,
    enabling targeted retrieval and filtering.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._collections: dict[str, Any] = {}

    def get_collection(self, language: str) -> Any:
        """
        Get or create the ChromaDB collection for a language.

        Args:
            language: Programming language (e.g., "python").

        Returns:
            The ChromaDB collection for the given language.
        """
        if language in self._collections:
            return self._collections[language]

        name = _collection_name(language)
        collection = self._client.get_or_create_collection(
            name=name,
            metadata={
                "language": language,
                "prefix": COLLECTION_NAME_PREFIX,
                "suffix": COLLECTION_SUFFIX,
            },
            embedding_function=None,  # We manage embeddings ourselves
        )
        self._collections[language] = collection
        return collection

    def upsert(
        self,
        language: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: Any = None,
    ) -> None:
        """
        Upsert chunks into a language-specific collection.

        Args:
            language: Programming language.
            ids: Chunk IDs.
            texts: Chunk texts (embedded content).
            metadatas: Per-chunk metadata dicts.
            embeddings: Pre-computed embedding vectors.
        """
        if not ids:
            return

        collection = self.get_collection(language)
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        language: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query a language-specific collection.

        Args:
            language: Programming language.
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Metadata filter.
            where_document: Document content filter.

        Returns:
            ChromaDB query result dict.
        """
        collection = self.get_collection(language)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

    def get_all_languages(self) -> list[str]:
        """
        List all languages that have collections.

        Returns:
            List of language names.
        """
        collections = self._client.list_collections()
        languages = []
        for col in collections:
            meta = col.metadata or {}
            if meta.get("prefix") == COLLECTION_NAME_PREFIX:
                lang = meta.get("language")
                if lang:
                    languages.append(lang)
        return languages

    def delete_file_chunks(self, language: str, file_path: str) -> None:
        """
        Delete all chunks associated with a specific file.

        Args:
            language: Programming language.
            file_path: Absolute path to the source file.
        """
        collection = self.get_collection(language)
        try:
            results = collection.get(
                where={"file_path": file_path},
                include=[],
            )
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
        except Exception:  # nosec: B110
            pass

    def reset(self) -> None:
        """Delete all collections (use with caution)."""
        self._collections.clear()
        for col in self._client.list_collections():
            try:
                self._client.delete_collection(col.name)
            except ChromaError:
                pass
        self._client.reset()
