"""Unit tests for CodeChunk."""

from __future__ import annotations

import pytest

from codebase_rag.core.chunk import CodeChunk


class TestCodeChunk:
    def test_valid_chunk(self):
        chunk = CodeChunk(
            text="def hello(): pass",
            language="python",
            node_type="function",
            file_path="/src/foo.py",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=19,
        )
        assert chunk.text == "def hello(): pass"
        assert chunk.language == "python"
        assert chunk.node_type == "function"
        assert chunk.start_line == 1
        assert chunk.end_line == 2

    def test_invalid_start_byte(self):
        with pytest.raises(ValueError, match="start_byte"):
            CodeChunk(
                text="def hello(): pass",
                language="python",
                node_type="function",
                file_path="/src/foo.py",
                start_line=2,
                end_line=1,
                start_byte=10,
                end_byte=5,
            )

    def test_invalid_start_line(self):
        with pytest.raises(ValueError, match="start_line"):
            CodeChunk(
                text="def hello(): pass",
                language="python",
                node_type="function",
                file_path="/src/foo.py",
                start_line=0,
                end_line=1,
                start_byte=0,
                end_byte=5,
            )

    def test_invalid_chunk_index(self):
        with pytest.raises(ValueError, match="chunk_index"):
            CodeChunk(
                text="def hello(): pass",
                language="python",
                node_type="function",
                file_path="/src/foo.py",
                start_line=1,
                end_line=1,
                start_byte=0,
                end_byte=5,
                chunk_index=-1,
            )

    def test_invalid_total_chunks(self):
        with pytest.raises(ValueError, match="total_chunks"):
            CodeChunk(
                text="def hello(): pass",
                language="python",
                node_type="function",
                file_path="/src/foo.py",
                start_line=1,
                end_line=1,
                start_byte=0,
                end_byte=5,
                total_chunks=0,
            )

    def test_to_metadata(self):
        chunk = CodeChunk(
            text="def hello(): pass",
            language="python",
            node_type="function",
            file_path="/src/foo.py",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=19,
            fully_qualified_name="hello",
            parent_name=None,
            chunk_index=0,
            total_chunks=1,
        )
        meta = chunk.to_metadata()
        assert meta["language"] == "python"
        assert meta["node_type"] == "function"
        assert meta["file_path"] == "/src/foo.py"
        assert meta["start_line"] == 1
        assert meta["end_line"] == 2
        assert meta["fully_qualified_name"] == "hello"

    def test_to_dict(self):
        chunk = CodeChunk(
            text="def hello(): pass",
            language="python",
            node_type="function",
            file_path="/src/foo.py",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=19,
        )
        d = chunk.to_dict()
        assert d["text"] == "def hello(): pass"
        assert d["language"] == "python"
        assert "id" in d
