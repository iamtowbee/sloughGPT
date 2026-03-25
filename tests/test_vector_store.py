"""
Tests for Vector Store Integration
"""

import pytest
import asyncio
import numpy as np
from domains.inference.vector_store import (
    InMemoryVectorStore,
    VectorEntry,
    QueryResult,
    VectorStoreFactory,
    simple_embed,
    VectorStoreType,
)


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""

    def test_initialization(self):
        """Test store initialization."""
        store = InMemoryVectorStore()
        assert store.entries == {}
        assert store.metric == "cosine"

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connection."""
        store = InMemoryVectorStore()
        result = await store.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_upsert(self):
        """Test document upsert."""
        store = InMemoryVectorStore()
        await store.connect()

        entries = [
            VectorEntry(
                id="1",
                vector=[0.1, 0.2, 0.3],
                text="Test document",
                metadata={"source": "test"},
            ),
            VectorEntry(
                id="2",
                vector=[0.4, 0.5, 0.6],
                text="Another document",
                metadata={"source": "test"},
            ),
        ]

        count = await store.upsert(entries)
        assert count == 2
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_query(self):
        """Test semantic search."""
        store = InMemoryVectorStore()
        await store.connect()

        entries = [
            VectorEntry(
                id="1",
                vector=[1.0, 0.0, 0.0],
                text="Python is a programming language",
                metadata={"source": "python"},
            ),
            VectorEntry(
                id="2",
                vector=[0.0, 1.0, 0.0],
                text="Machine learning is AI",
                metadata={"source": "ml"},
            ),
            VectorEntry(
                id="3",
                vector=[0.0, 0.0, 1.0],
                text="Deep learning uses neural networks",
                metadata={"source": "dl"},
            ),
        ]

        await store.upsert(entries)

        results = await store.query(
            vector=[1.0, 0.0, 0.0],
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].id == "1"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test document deletion."""
        store = InMemoryVectorStore()
        await store.connect()

        entries = [
            VectorEntry(id="1", vector=[0.1], text="Doc 1"),
            VectorEntry(id="2", vector=[0.2], text="Doc 2"),
        ]
        await store.upsert(entries)

        await store.delete(["1"])
        assert await store.count() == 1

        results = await store.query(vector=[0.1], top_k=5)
        assert all(r.id != "1" for r in results)

    @pytest.mark.asyncio
    async def test_filter_metadata(self):
        """Test metadata filtering."""
        store = InMemoryVectorStore()
        await store.connect()

        entries = [
            VectorEntry(id="1", vector=[0.1], text="Doc 1", metadata={"type": "a"}),
            VectorEntry(id="2", vector=[0.2], text="Doc 2", metadata={"type": "b"}),
            VectorEntry(id="3", vector=[0.3], text="Doc 3", metadata={"type": "a"}),
        ]
        await store.upsert(entries)

        results = await store.query(
            vector=[0.1],
            top_k=5,
            filter_metadata={"type": "a"},
        )

        assert len(results) == 2
        assert all(r.metadata.get("type") == "a" for r in results)


class TestSimpleEmbed:
    """Tests for simple embedding function."""

    def test_embed_dimension(self):
        """Test embedding dimension."""
        vec = simple_embed("test text", dimension=384)
        assert len(vec) == 384

    def test_embed_normalized(self):
        """Test embedding is normalized."""
        vec = simple_embed("test text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_consistent(self):
        """Test same text gives same embedding."""
        vec1 = simple_embed("hello world")
        vec2 = simple_embed("hello world")
        assert np.allclose(vec1, vec2)

    def test_embed_different_texts(self):
        """Test different texts give different embeddings."""
        vec1 = simple_embed("python programming")
        vec2 = simple_embed("machine learning")
        assert not np.allclose(vec1, vec2)


class TestVectorStoreFactory:
    """Tests for vector store factory."""

    def test_create_in_memory(self):
        """Test creating in-memory store."""
        store = VectorStoreFactory.create("in_memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_create_with_enum(self):
        """Test creating store with enum type."""
        store = VectorStoreFactory.create(VectorStoreType.IN_MEMORY)
        assert isinstance(store, InMemoryVectorStore)

    def test_create_invalid_type(self):
        """Test creating with invalid type falls back to in-memory."""
        store = VectorStoreFactory.create("invalid_type")
        assert isinstance(store, InMemoryVectorStore)


class TestVectorEntry:
    """Tests for VectorEntry dataclass."""

    def test_vector_entry_creation(self):
        """Test VectorEntry creation."""
        entry = VectorEntry(
            id="test_id",
            vector=[0.1, 0.2, 0.3],
            text="Test text",
            metadata={"key": "value"},
        )

        assert entry.id == "test_id"
        assert entry.vector == [0.1, 0.2, 0.3]
        assert entry.text == "Test text"
        assert entry.metadata == {"key": "value"}

    def test_vector_entry_default_metadata(self):
        """Test default metadata."""
        entry = VectorEntry(id="1", vector=[0.1], text="test")
        assert entry.metadata == {}


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test QueryResult creation."""
        result = QueryResult(
            id="test_id",
            score=0.95,
            text="Result text",
            metadata={"source": "test"},
        )

        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.text == "Result text"
        assert result.metadata == {"source": "test"}
