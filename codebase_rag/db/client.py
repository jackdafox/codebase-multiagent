"""ChromaDB client singleton."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    import chromadb

_CLIENT_INSTANCE: "chromadb.PersistentClient | None" = None
_CLIENT_LOCK = threading.Lock()


def get_client(
    persist_directory: str = "./chroma_db",
    settings: Settings | None = None,
) -> "chromadb.PersistentClient":
    """
    Get or create the singleton ChromaDB client.

    Args:
        persist_directory: Directory for ChromaDB persistence.
        settings: Optional ChromaDB settings override.

    Returns:
        A persistent ChromaDB client instance.
    """
    global _CLIENT_INSTANCE

    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE

    with _CLIENT_LOCK:
        if _CLIENT_INSTANCE is not None:
            return _CLIENT_INSTANCE

        _CLIENT_INSTANCE = chromadb.PersistentClient(
            path=persist_directory,
            settings=settings or Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        return _CLIENT_INSTANCE


def reset_client() -> None:
    """Reset the singleton client (for testing)."""
    global _CLIENT_INSTANCE
    with _CLIENT_LOCK:
        _CLIENT_INSTANCE = None
