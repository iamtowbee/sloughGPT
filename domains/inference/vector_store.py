"""
Production-Grade Vector Store Integration

Supports:
- Pinecone (cloud-hosted)
- Weaviate (self-hosted or cloud)
- ChromaDB (local/simplest)
- In-memory fallback for development

Usage:
    from domains.inference.vector_store import VectorStore, PineconeStore, WeaviateStore
    
    # Pinecone
    store = PineconeStore(api_key="...", index="production")
    await store.connect()
    await store.upsert([{"id": "1", "vector": [...], "text": "..."}])
    results = await store.query(vector=[...], top_k=5)
    
    # Weaviate
    store = WeaviateStore(url="http://localhost:8080")
    await store.connect()
    await store.upsert([{"id": "1", "vector": [...], "text": "..."}])
    results = await store.query(vector=[...], top_k=5)
"""

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from enum import Enum

import numpy as np


class VectorStoreType(Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMADB = "chromadb"
    IN_MEMORY = "in_memory"


@dataclass
class VectorEntry:
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def upsert(self, entries: List[VectorEntry]) -> int:
        pass

    @abstractmethod
    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass


class InMemoryVectorStore(VectorStore):
    def __init__(self, metric: str = "cosine"):
        self.entries: Dict[str, VectorEntry] = {}
        self.metric = metric

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self.entries.clear()

    async def upsert(self, entries: List[VectorEntry]) -> int:
        count = 0
        for entry in entries:
            self.entries[entry.id] = entry
            count += 1
        return count

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.entries:
            return []

        query_vec = np.array(vector)
        scores = []

        for entry in self.entries.values():
            if filter_metadata:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            entry_vec = np.array(entry.vector)
            if self.metric == "cosine":
                score = float(np.dot(query_vec, entry_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-8
                ))
            elif self.metric == "euclidean":
                score = float(-np.linalg.norm(query_vec - entry_vec))
            else:
                score = float(np.dot(query_vec, entry_vec))

            scores.append((entry, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            QueryResult(
                id=e.id,
                score=s,
                text=e.text,
                metadata=e.metadata,
            )
            for e, s in scores[:top_k]
        ]

    async def delete(self, ids: List[str]) -> bool:
        for id in ids:
            self.entries.pop(id, None)
        return True

    async def count(self) -> int:
        return len(self.entries)


class PineconeStore(VectorStore):
    def __init__(
        self,
        api_key: Optional[str] = None,
        index: str = "sloughgpt",
        environment: str = "us-east-1",
        metric: str = "cosine",
        dimension: int = 768,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index
        self.environment = environment
        self.metric = metric
        self.dimension = dimension
        self.index = None
        self._client = None

    async def connect(self) -> bool:
        try:
            from pinecone import Pinecone
            self._client = Pinecone(api_key=self.api_key)
            
            existing = [i.name for i in self._client.list_indexes()]
            
            if self.index_name not in existing:
                self._client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                )
            
            self.index = self._client.Index(self.index_name)
            return True
        except ImportError:
            raise ImportError("pip install pinecone-client")
        except Exception as e:
            print(f"Pinecone connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.index = None
        self._client = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        vectors = [
            {
                "id": e.id,
                "values": e.vector,
                "metadata": {"text": e.text, **e.metadata},
            }
            for e in entries
        ]

        self.index.upsert(vectors=vectors)
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_metadata,
        )

        return [
            QueryResult(
                id=r["id"],
                score=r["score"],
                text=r["metadata"].get("text", ""),
                metadata={k: v for k, v in r["metadata"].items() if k != "text"},
            )
            for r in results.get("matches", [])
        ]

    async def delete(self, ids: List[str]) -> bool:
        if not self.index:
            return False
        self.index.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.index:
            return 0
        return self.index.describe_index_stats().get("total_vector_count", 0)


class WeaviateStore(VectorStore):
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "Document",
        dimension: int = 768,
    ):
        self.url = url
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.class_name = class_name
        self.dimension = dimension
        self.client = None

    async def connect(self) -> bool:
        try:
            import weaviate
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
            
            self.client = weaviate.Client(
                url=self.url,
                auth_credentials=auth_config,
            )
            
            if not self.client.schema.exists(self.class_name):
                self.client.schema.create_class({
                    "class": self.class_name,
                    "vectorizer": "none",
                    "moduleConfig": {
                        "none": {
                            "vectorizeClassName": False,
                        },
                    },
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "chunk_id", "dataType": ["text"]},
                    ],
                })
            
            return True
        except ImportError:
            raise ImportError("pip install weaviate-client")
        except Exception as e:
            print(f"Weaviate connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.client = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.client:
            raise RuntimeError("Not connected to Weaviate")

        with self.client.batch(batch_size=100) as batch:
            for entry in entries:
                batch.add_data_object(
                    data_object={
                        "text": entry.text,
                        "source": entry.metadata.get("source", ""),
                        "chunk_id": entry.id,
                    },
                    class_name=self.class_name,
                    vector=entry.vector,
                )

        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.client:
            raise RuntimeError("Not connected to Weaviate")

        where_filter = None
        if filter_metadata:
            where_filter = {
                "path": ["source"],
                "operator": "Equal",
                "valueString": filter_metadata.get("source", ""),
            }

        results = self.client.query.get(
            self.class_name, ["text", "source", "chunk_id"]
        ).with_near_vector(
            {"vector": vector}
        ).with_limit(top_k)

        if where_filter:
            results = results.with_where(where_filter)

        response = results.do()
        matches = response.get("data", {}).get("Get", {}).get(self.class_name, [])

        return [
            QueryResult(
                id=obj.get("chunk_id", ""),
                score=1.0,
                text=obj.get("text", ""),
                metadata={"source": obj.get("source", "")},
            )
            for obj in matches
        ]

    async def delete(self, ids: List[str]) -> bool:
        if not self.client:
            return False

        for id in ids:
            self.client.data_object.delete(id, class_name=self.class_name)
        return True

    async def count(self) -> int:
        if not self.client:
            return 0
        result = self.client.query.aggregate(self.class_name).with_meta_count().do()
        return result.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)


class ChromaStore(VectorStore):
    def __init__(
        self,
        persist_directory: str = "./vector_store",
        collection_name: str = "documents",
        distance_metric: str = "cosine",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.client = None
        self.collection = None

    async def connect(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            return True
        except ImportError:
            raise ImportError("pip install chromadb")
        except Exception as e:
            print(f"ChromaDB connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.client = None
        self.collection = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")

        self.collection.upsert(
            ids=[e.id for e in entries],
            embeddings=[e.vector for e in entries],
            documents=[e.text for e in entries],
            metadatas=[e.metadata for e in entries],
        )
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter_metadata,
        )

        entries = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                entries.append(QueryResult(
                    id=id,
                    score=1.0 - (results["distances"][0][i] if results.get("distances") else 0),
                    text=results["documents"][0][i] if results.get("documents") else "",
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                ))
        return entries

    async def delete(self, ids: List[str]) -> bool:
        if not self.collection:
            return False
        self.collection.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.collection:
            return 0
        return self.collection.count()


class VectorStoreFactory:
    @staticmethod
    def create(
        store_type: Union[str, VectorStoreType],
        **kwargs,
    ) -> VectorStore:
        if isinstance(store_type, str):
            try:
                store_type = VectorStoreType(store_type.lower())
            except ValueError:
                return InMemoryVectorStore(**kwargs)

        stores = {
            VectorStoreType.PINECONE: PineconeStore,
            VectorStoreType.WEAVIATE: WeaviateStore,
            VectorStoreType.CHROMADB: ChromaStore,
            VectorStoreType.IN_MEMORY: InMemoryVectorStore,
        }

        store_class = stores.get(store_type, InMemoryVectorStore)
        return store_class(**kwargs)


async def create_vector_store(
    provider: Optional[str] = None,
    **kwargs,
) -> VectorStore:
    provider = provider or os.getenv("VECTOR_STORE_PROVIDER", "in_memory")
    
    store = VectorStoreFactory.create(provider, **kwargs)
    await store.connect()
    return store


def simple_embed(text: str, dimension: int = 768) -> List[float]:
    vec = np.zeros(dimension)
    words = text.lower().split()
    
    for i, word in enumerate(words[:dimension]):
        word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        vec[i % dimension] += np.sin(word_hash * (i + 1) * 0.1)
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


__all__ = [
    "VectorStore",
    "VectorEntry",
    "QueryResult",
    "VectorStoreType",
    "InMemoryVectorStore",
    "PineconeStore",
    "WeaviateStore",
    "ChromaStore",
    "VectorStoreFactory",
    "create_vector_store",
    "simple_embed",
]
