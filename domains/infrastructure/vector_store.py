"""
Vector Store - Ported from recovered hauls_store.py
A lightweight vector database for semantic search and RAG.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class Document:
    """Document structure for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


class VectorIndex:
    """Simple vector index for semantic search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Dict[str, np.ndarray] = {}
        self.doc_mapping: Dict[str, int] = {}
    
    def add_document(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add document to index"""
        self.index[doc_id] = embedding
        self.doc_mapping[doc_id] = len(self.doc_mapping)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        if not self.index:
            return []
        
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for doc_id, doc_embedding in self.index.items():
            doc_norm = np.linalg.norm(doc_embedding)
            if doc_norm > 0 and query_norm > 0:
                similarity = float(np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm))
            else:
                similarity = 0.0
            similarities.append((doc_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def __len__(self) -> int:
        return len(self.index)


class VectorStore:
    """Vector database with persistent storage"""
    
    def __init__(self, db_path: str = "data/vector_store.db", dimension: int = 384):
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.index = VectorIndex(dimension)
        self.documents: Dict[str, Document] = {}
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                embedding BLOB,
                created_at TEXT
            )
        """)
        self.conn.commit()
    
    def add_document(self, doc: Document) -> None:
        """Add a document to the store"""
        self.documents[doc.id] = doc
        
        if doc.embedding is not None:
            self.index.add_document(doc.id, doc.embedding)
        
        self.conn.execute("""
            INSERT OR REPLACE INTO documents (id, content, metadata, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc.id,
            doc.content,
            json.dumps(doc.metadata),
            pickle.dumps(doc.embedding) if doc.embedding is not None else None,
            doc.created_at
        ))
        self.conn.commit()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        results = self.index.search(query_embedding, top_k)
        
        return [
            (self.documents[doc_id], score)
            for doc_id, score in results
            if doc_id in self.documents
        ]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.index.index:
                del self.index.index[doc_id]
            
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.commit()
            return True
        return False
    
    def close(self) -> None:
        """Close database connection"""
        self.conn.close()


def simple_embedding(text: str, dimension: int = 384) -> np.ndarray:
    """Create a simple hash-based embedding"""
    import hashlib
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2**32))
    return np.random.randn(dimension).astype(np.float32)


__all__ = ["VectorStore", "VectorIndex", "Document", "simple_embedding"]
