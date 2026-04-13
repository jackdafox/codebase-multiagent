"""Unit tests for db/collections.py with mocked ChromaDB."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebase_rag.db.collections import CollectionManager, _collection_name


class TestCollectionName:
    def test_python_collection_name(self):
        assert _collection_name("python") == "codebase_python_chunks"

    def test_go_collection_name(self):
        assert _collection_name("go") == "codebase_go_chunks"

    def test_rust_collection_name(self):
        assert _collection_name("rust") == "codebase_rust_chunks"


class TestCollectionManager:
    """Tests with mocked ChromaDB client."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_client):
        return CollectionManager(mock_client)

    def test_get_collection_creates_once(self, mock_client):
        manager = CollectionManager(mock_client)
        coll = manager.get_collection("python")
        assert coll is not None
        assert mock_client.get_or_create_collection.call_count == 1
        # Second call returns cached
        coll2 = manager.get_collection("python")
        assert coll2 is coll
        assert mock_client.get_or_create_collection.call_count == 1

    def test_get_collection_different_languages(self, mock_client):
        manager = CollectionManager(mock_client)
        manager.get_collection("python")
        manager.get_collection("go")
        assert mock_client.get_or_create_collection.call_count == 2

    def test_upsert_empty_ids_skips(self, mock_client):
        manager = CollectionManager(mock_client)
        manager.upsert("python", [], [], [])
        mock_client.get_or_create_collection.assert_not_called()

    def test_upsert_calls_collection_upsert(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll

        manager.upsert(
            language="python",
            ids=["id1", "id2"],
            texts=["def foo(): pass", "def bar(): pass"],
            metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        mock_coll.upsert.assert_called_once()
        call_kwargs = mock_coll.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["id1", "id2"]
        assert call_kwargs["documents"] == ["def foo(): pass", "def bar(): pass"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    def test_query_calls_collection_query(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll = MagicMock()
        mock_coll.query.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.get_or_create_collection.return_value = mock_coll

        manager.query("python", query_embedding=[0.1, 0.2], n_results=5)
        mock_coll.query.assert_called_once()
        call_kwargs = mock_coll.query.call_args.kwargs
        assert call_kwargs["query_embeddings"] == [[0.1, 0.2]]
        assert call_kwargs["n_results"] == 5

    def test_query_with_metadata_filter(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll = MagicMock()
        mock_coll.query.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.get_or_create_collection.return_value = mock_coll

        manager.query(
            "python",
            query_embedding=[0.1],
            n_results=3,
            where={"file_path": "/src/a.py"},
        )
        call_kwargs = mock_coll.query.call_args.kwargs
        assert call_kwargs["where"] == {"file_path": "/src/a.py"}

    def test_delete_file_chunks(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": ["id1", "id2"]}
        mock_client.get_or_create_collection.return_value = mock_coll

        manager.delete_file_chunks("python", "/src/a.py")
        mock_coll.get.assert_called_once()
        mock_coll.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_delete_file_chunks_no_matches(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_coll

        manager.delete_file_chunks("python", "/nonexistent.py")
        mock_coll.delete.assert_not_called()

    def test_reset_clears_cache_and_deletes(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_coll1 = MagicMock()
        mock_coll1.name = "codebase_python_chunks"
        mock_client.get_or_create_collection.return_value = mock_coll1
        manager.get_collection("python")
        mock_client.list_collections.return_value = [mock_coll1]

        manager.reset()
        mock_client.delete_collection.assert_called()
        mock_client.reset.assert_called_once()

    def test_get_all_languages(self, mock_client):
        manager = CollectionManager(mock_client)
        mock_col1 = MagicMock()
        mock_col1.metadata = {"prefix": "codebase", "language": "python"}
        mock_col2 = MagicMock()
        mock_col2.metadata = {"prefix": "codebase", "language": "go"}
        mock_col3 = MagicMock()
        mock_col3.metadata = {"prefix": "other", "language": "rust"}
        mock_col4 = MagicMock()
        mock_col4.metadata = {}
        mock_client.list_collections.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]

        langs = manager.get_all_languages()
        assert "python" in langs
        assert "go" in langs
        assert "rust" not in langs  # wrong prefix
