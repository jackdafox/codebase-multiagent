"""Unit tests for CodeChunker."""

from __future__ import annotations


import pytest

from codebase_rag.core.chunker import CodeChunker


class TestCodeChunker:
    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    def test_chunk_python_file(self, chunker, sample_python_file):
        content = sample_python_file.read_bytes()
        chunks = chunker.chunk_file(str(sample_python_file), content, "python")

        assert len(chunks) >= 2  # At least hello function + Greeter class

        # Check chunk structure
        for chunk in chunks:
            assert chunk.text
            assert chunk.language == "python"
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.file_path == str(sample_python_file)
            assert chunk.node_type in ("function", "class", "method", "file")

    def test_chunk_javascript_file(self, chunker, sample_js_file):
        content = sample_js_file.read_bytes()
        chunks = chunker.chunk_file(str(sample_js_file), content, "javascript")

        assert len(chunks) >= 2  # At least hello function + Greeter class

        for chunk in chunks:
            assert chunk.text
            assert chunk.language == "javascript"

    def test_chunk_go_file(self, chunker, sample_go_file):
        content = sample_go_file.read_bytes()
        chunks = chunker.chunk_file(str(sample_go_file), content, "go")

        assert len(chunks) >= 2  # At least Hello function + Greeter type

        for chunk in chunks:
            assert chunk.text
            assert chunk.language == "go"

    def test_chunks_ordered_by_position(self, chunker, sample_python_file):
        content = sample_python_file.read_bytes()
        chunks = chunker.chunk_file(str(sample_python_file), content, "python")

        # Chunks should be in source order
        for i in range(len(chunks) - 1):
            assert chunks[i].start_byte <= chunks[i + 1].start_byte

    def test_chunk_index_assigned(self, chunker, sample_python_file):
        content = sample_python_file.read_bytes()
        chunks = chunker.chunk_file(str(sample_python_file), content, "python")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_large_file_fallback(self, chunker, temp_dir):
        # Create a file larger than MAX_FILE_SIZE
        large_file = temp_dir / "large.py"
        large_file.write_bytes(b"# " + b"x" * 1_000_001)

        content = large_file.read_bytes()
        chunks = chunker.chunk_file(str(large_file), content, "python")

        # Should fall back to single chunk
        assert len(chunks) == 1
        assert chunks[0].node_type == "file"

    def test_invalid_utf8(self, chunker, temp_dir):
        # File with invalid UTF-8
        bad_file = temp_dir / "bad.py"
        bad_file.write_bytes(b"# coding: latin-1\n" + bytes(range(256)))

        content = bad_file.read_bytes()
        chunks = chunker.chunk_file(str(bad_file), content, "python")

        # Should not crash, should return some chunks
        assert isinstance(chunks, list)
