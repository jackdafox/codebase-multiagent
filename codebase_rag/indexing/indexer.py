"""CodebaseIndexer -- orchestrates indexing source code into ChromaDB."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from codebase_rag.core.chunker import CodeChunker
from codebase_rag.core.chunk import CodeChunk
from codebase_rag.core.language import (
    detect_language,
    get_glob_patterns,
    is_supported_language,
)
from codebase_rag.db.client import get_client
from codebase_rag.db.collections import CollectionManager
from codebase_rag.embeddings.factory import (
    DEFAULT_MODEL,
    get_embedding_function,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
DEFAULT_WORKERS = 4


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    files_indexed: int = 0
    chunks_created: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    languages_found: set[str] = field(default_factory=set)

    @property
    def had_errors(self) -> bool:
        return len(self.errors) > 0


class CodebaseIndexer:
    """
    Indexes a codebase directory into ChromaDB per-language collections.

    Walk a directory tree, detect source files by language, parse them
    with tree-sitter, extract semantic chunks, embed with OpenAI,
    and store in ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = DEFAULT_MODEL,
        embedding_dimensions: int | None = None,
        embedding_api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_WORKERS,
        collection_prefix: str = "codebase",
    ) -> None:
        """
        Initialize the indexer.

        Args:
            persist_directory: ChromaDB persistence directory.
            embedding_model: OpenAI embedding model name.
            embedding_dimensions: Embedding dimensions (for text-embedding-3).
            embedding_api_key: OpenAI API key override.
            batch_size: Files to process per embedding batch.
            num_workers: Parallel workers for parsing (future).
            collection_prefix: Prefix for ChromaDB collection names.
        """
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._batch_size = batch_size

        self._client = get_client(persist_directory)
        self._collections = CollectionManager(self._client)
        self._chunker = CodeChunker()
        self._embedding_factory = get_embedding_function(
            model=embedding_model,
            dimensions=embedding_dimensions,
            api_key=embedding_api_key,
        )

    def index(
        self,
        root_path: str | Path,
        languages: list[str] | None = None,
        glob_patterns: dict[str, str] | None = None,
        incremental: bool = False,
        show_progress: bool = True,
    ) -> IndexResult:
        """
        Index all supported files under root_path.

        Args:
            root_path: Directory to index recursively.
            languages: Restrict to specific languages (None = all supported).
            glob_patterns: Override glob patterns per language.
            incremental: Only index files changed since last run.
            show_progress: Display progress bar.

        Returns:
            IndexResult with counts, errors, and timing.
        """
        start_time = time.time()
        root = Path(root_path).resolve()

        if not root.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")

        # Build language -> glob pattern map
        patterns = get_glob_patterns(languages, glob_patterns)

        # Discover files by language
        files_by_lang: dict[str, list[Path]] = {}
        for lang, pattern in patterns.items():
            matches = list(root.glob(pattern))
            # Filter to only actual files under root
            matches = [p for p in matches if p.is_file() and str(p).startswith(str(root))]
            if matches:
                files_by_lang[lang] = matches

        # Also detect files by extension that might not match glob
        self._detect_additional_files(root, files_by_lang)

        result = IndexResult()

        # Process each language
        for lang, file_paths in files_by_lang.items():
            if not is_supported_language(lang):
                logger.warning("Unsupported language: %s", lang)
                continue

            lang_chunks: list[CodeChunk] = []
            for path in tqdm(file_paths, desc=f"Parsing {lang}", disable=not show_progress):
                try:
                    chunks = self._index_file(path, lang)
                    if chunks:
                        lang_chunks.extend(chunks)
                        result.chunks_created += len(chunks)
                        result.files_indexed += 1
                        result.languages_found.add(lang)
                except Exception as e:
                    logger.error("Error indexing %s: %s", path, e)
                    result.errors.append({
                        "file": str(path),
                        "language": lang,
                        "error": str(e),
                    })

            # Batch upsert chunks to ChromaDB
            if lang_chunks:
                self._upsert_chunks({lang: lang_chunks})

        result.duration_seconds = time.time() - start_time
        return result

    def _detect_additional_files(
        self,
        root: Path,
        files_by_lang: dict[str, list[Path]],
    ) -> None:
        """Detect files by extension scanning for any missed by glob."""
        # Walk the tree and detect by extension
        seen: set[str] = set()
        for lang_files in files_by_lang.values():
            for f in lang_files:
                seen.add(str(f))

        for path in root.rglob("*"):
            if path.is_file() and str(path) not in seen:
                lang = detect_language(path)
                if lang and is_supported_language(lang):
                    if lang not in files_by_lang:
                        files_by_lang[lang] = []
                    files_by_lang[lang].append(path)
                    seen.add(str(path))

    def _index_file(self, path: Path, language: str) -> list[CodeChunk]:
        """Parse a single file and return its chunks."""
        try:
            content = path.read_bytes()
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return []

        return self._chunker.chunk_file(str(path), content, language)

    def _upsert_chunks(self, chunks_by_lang: dict[str, list[CodeChunk]]) -> None:
        """Batch embed and upsert chunks to ChromaDB per language."""
        for lang, chunks in chunks_by_lang.items():
            if not chunks:
                continue

            # Prepare batch
            ids = [c.id for c in chunks]
            texts = [c.text for c in chunks]
            metadatas = [c.to_metadata() for c in chunks]

            # Embed in batches
            batch_size = self._batch_size
            for i in range(0, len(texts), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]

                batch_embeddings = self._embedding_factory.embed_texts(batch_texts)

                self._collections.upsert(
                    language=lang,
                    ids=batch_ids,
                    texts=batch_texts,
                    metadatas=batch_metas,
                    embeddings=batch_embeddings,
                )

    def index_file(self, file_path: str, language: str | None = None) -> list[CodeChunk]:
        """
        Index a single file.

        Args:
            file_path: Path to the source file.
            language: Language override (auto-detected if None).

        Returns:
            List of CodeChunks created.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        lang = language or detect_language(path)
        if not lang:
            raise ValueError(f"Could not detect language for: {file_path}")

        chunks = self._index_file(path, lang)
        if chunks:
            chunks_by_file = {lang: chunks}
            self._upsert_chunks(chunks_by_file)
        return chunks

    def delete_file(self, file_path: str, language: str) -> None:
        """
        Remove all chunks for a file from the index.

        Args:
            file_path: Absolute path to the source file.
            language: Programming language.
        """
        self._collections.delete_file_chunks(language, file_path)

    def update_file(self, file_path: str, language: str | None = None) -> list[CodeChunk]:
        """
        Remove old chunks for a file and re-index.

        Args:
            file_path: Path to the source file.
            language: Language override.

        Returns:
            List of new CodeChunks.
        """
        lang = language or detect_language(file_path)
        if not lang:
            raise ValueError(f"Could not detect language for: {file_path}")

        self.delete_file(file_path, lang)
        return self.index_file(file_path, lang)
