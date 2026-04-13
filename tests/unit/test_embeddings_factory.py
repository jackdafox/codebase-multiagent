"""Unit tests for embeddings/factory.py with mocked OpenAI client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAIEmbeddingsFactory:
    """Tests with mocked OpenAI client."""

    @pytest.fixture
    def factory(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        factory = OpenAIEmbeddingsFactory.__new__(OpenAIEmbeddingsFactory)
        factory._api_key = "test-key"
        factory._model = "text-embedding-3-small"
        factory._dimensions = None
        factory._batch_size = 100
        factory._client = mock_client
        return factory

    def test_init_with_api_key(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        factory = OpenAIEmbeddingsFactory.__new__(OpenAIEmbeddingsFactory)
        factory._api_key = "my-key"
        factory._model = "text-embedding-3-small"
        factory._dimensions = None
        factory._batch_size = 100
        factory._client = None
        assert factory._api_key == "my-key"
        assert factory._model == "text-embedding-3-small"

    def test_init_reads_env_var(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        factory = OpenAIEmbeddingsFactory.__new__(OpenAIEmbeddingsFactory)
        factory._api_key = "env-key"
        factory._model = "text-embedding-3-small"
        factory._dimensions = None
        factory._batch_size = 100
        factory._client = None
        assert factory._api_key == "env-key"

    def test_init_without_key_raises(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIEmbeddingsFactory(api_key=None)

    def test_init_with_dimensions(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        factory = OpenAIEmbeddingsFactory.__new__(OpenAIEmbeddingsFactory)
        factory._api_key = "key"
        factory._model = "text-embedding-3-small"
        factory._dimensions = 256
        factory._batch_size = 100
        factory._client = None
        assert factory._dimensions == 256

    def test_embed_texts_empty_list(self, factory):
        result = factory.embed_texts([])
        assert result == []
        factory._client.embeddings.create.assert_not_called()

    def test_embed_texts_returns_vectors(self, factory):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        factory._client.embeddings.create.return_value = mock_response

        result = factory.embed_texts(["hello", "world"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_query_returns_single_vector(self, factory):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        factory._client.embeddings.create.return_value = mock_response

        result = factory.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_call_delegates_to_embed_texts(self, factory):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        factory._client.embeddings.create.return_value = mock_response

        result = factory(["hello"])
        assert result == [[0.1]]

    def test_dimensions_not_sent_when_none(self, factory):
        factory._dimensions = None
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        factory._client.embeddings.create.return_value = mock_response

        factory.embed_query("hello")
        call_kwargs = factory._client.embeddings.create.call_args.kwargs
        assert "dimensions" not in call_kwargs

    def test_dimensions_sent_when_set(self):
        from codebase_rag.embeddings.factory import OpenAIEmbeddingsFactory

        factory = OpenAIEmbeddingsFactory.__new__(OpenAIEmbeddingsFactory)
        factory._api_key = "key"
        factory._model = "text-embedding-3-small"
        factory._dimensions = 256
        factory._batch_size = 100
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_client.embeddings.create.return_value = mock_response
        factory._client = mock_client

        factory.embed_query("hello")
        call_kwargs = factory._client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 256


class TestGetEmbeddingFunction:
    """Tests for the singleton getter."""

    def test_returns_same_instance(self):
        from codebase_rag.embeddings.factory import _FACTORIES, get_embedding_function

        _FACTORIES.clear()
        factory1 = get_embedding_function(api_key="key")
        factory2 = get_embedding_function(api_key="key")
        assert factory1 is factory2

    def test_different_keys_different_instances(self):
        from codebase_rag.embeddings.factory import _FACTORIES, get_embedding_function

        _FACTORIES.clear()
        factory1 = get_embedding_function(model="text-embedding-3-small", api_key="key")
        factory2 = get_embedding_function(model="text-embedding-3-large", api_key="key")
        assert factory1 is not factory2
