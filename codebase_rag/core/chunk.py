"""CodeChunk dataclass -- the fundamental data unit for the RAG system."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    """
    Represents a parsed code unit extracted from a source file.

    Attributes:
        id: Unique identifier (UUID4).
        text: Raw source code of this chunk.
        language: Programming language (e.g., "python", "javascript").
        node_type: AST node type (e.g., "function", "class", "method").
        file_path: Absolute path to the source file.
        start_line: 1-indexed start line number.
        end_line: 1-indexed end line number.
        start_byte: Byte offset in file where chunk starts.
        end_byte: Byte offset in file where chunk ends.
        fully_qualified_name: Full semantic name (e.g., "module.ClassName.method_name").
        parent_name: Name of enclosing class or function, if any.
        chunk_index: Position of this chunk within the file's ordered chunks.
        total_chunks: Total number of chunks extracted from this file.
        docstring: Extracted docstring or leading comment, if any.
        signature: Function/method signature, if applicable.
    """

    text: str
    language: str
    node_type: str
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    fully_qualified_name: str | None = None
    parent_name: str | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    docstring: str | None = None
    signature: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> dict:
        """Serialize non-text fields for ChromaDB metadata storage."""
        return {
            "id": self.id,
            "language": self.language,
            "node_type": self.node_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "fully_qualified_name": self.fully_qualified_name,
            "parent_name": self.parent_name,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "docstring": self.docstring or "",
            "signature": self.signature or "",
            "file_hash": self.file_hash,
        }

    @property
    def file_hash(self) -> str:
        """SHA256 hash of the source file content, if file is readable."""
        try:
            with open(self.file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return ""

    def to_dict(self) -> dict:
        """Full serialization including text."""
        return {
            "id": self.id,
            "text": self.text,
            **self.to_metadata(),
        }

    def __post_init__(self) -> None:
        # Ensure IDs are stable within a session for the same content
        if self.start_byte > self.end_byte:
            raise ValueError(
                f"start_byte ({self.start_byte}) must be <= end_byte ({self.end_byte})"
            )
        if self.start_line < 1:
            raise ValueError(f"start_line ({self.start_line}) must be >= 1")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index ({self.chunk_index}) must be >= 0")
        if self.total_chunks < 1:
            raise ValueError(f"total_chunks ({self.total_chunks}) must be >= 1")
