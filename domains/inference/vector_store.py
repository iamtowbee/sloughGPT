"""
Production-Grade Vector Store - Pinecone Only

Usage:
    from domains.inference.vector_store import PineconeVectorStore
    
    store = PineconeVectorStore(
        api_key="your-api-key",
        index_name="production"
    )
    await store.connect()
    await store.upsert([{"id": "1", "vector": [...], "text": "..."}])
    results = await store.query(vector=[...], top_k=5)
"""

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


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


class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "sloughgpt",
        environment: str = "us-east-1",
        dimension: int = 768,
        metric: str = "cosine",
        host: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.environment = environment
        self.dimension = dimension
        self.metric = metric
        self.host = host
        self.index = None
        self.serverless_spec = None
        self.pod_spec = None

    async def connect(self) -> bool:
        try:
            from pinecone import Pinecone, ServerlessSpec, PodSpec

            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is required")

            self.client = Pinecone(api_key=self.api_key)

            if self.index_name not in [idx.name for idx in self.client.list_indexes()]:
                if self.environment in ["us-east-1", "us-west-2", "eu-west-1"]:
                    self.serverless_spec = ServerlessSpec(
                        cloud=self.environment.split("-")[0].upper(),
                        region=self.environment
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.serverless_spec,
                    )
                else:
                    self.pod_spec = PodSpec(
                        environment=self.environment,
                        replicas=1,
                        shards=1
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.pod_spec,
                    )

            self.index = self.client.Index(self.index_name)
            return True
        except ImportError:
            raise ImportError("pip install pinecone-client")
        except Exception as e:
            print(f"Pinecone connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self.index = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        vectors = []
        for entry in entries:
            vectors.append({
                "id": entry.id,
                "values": entry.vector,
                "metadata": {
                    "text": entry.text,
                    **entry.metadata
                }
            })

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

        query_params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }

        if filter_metadata:
            query_params["filter"] = filter_metadata

        results = self.index.query(**query_params)

        return [
            QueryResult(
                id=match["id"],
                score=match.get("score", 0.0),
                text=match.get("metadata", {}).get("text", ""),
                metadata={k: v for k, v in match.get("metadata", {}).items() if k != "text"},
            )
            for match in results.get("matches", [])
        ]

    async def delete(self, ids: List[str]) -> bool:
        if not self.index:
            return False
        self.index.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.index:
            return 0
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)


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
    "PineconeVectorStore",
    "simple_embed",
]
