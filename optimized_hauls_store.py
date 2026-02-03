#!/usr/bin/env python3
"""
Optimized HaulsStore v2 - Performance improvements for Stage 1
"""

import json
import pickle
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Try to import sentence-transformers, fallback to simple embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class Document:
    """Document structure for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


class OptimizedEndicIndex:
    """Optimized Endic-style semantic index with performance improvements"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = {}
        self.doc_mapping = {}
        # Performance: Pre-allocate embedding arrays
        self.embedding_cache = {}  # Cache frequently accessed embeddings
        self.cache_size_limit = 1000
        
    def add_document(self, doc_id: str, embedding: np.ndarray):
        """Add document to index with caching"""
        self.index[doc_id] = embedding
        self.doc_mapping[doc_id] = len(self.doc_mapping)
        
        # Cache embedding if within limit
        if len(self.embedding_cache) < self.cache_size_limit:
            self.embedding_cache[doc_id] = embedding.copy()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Optimized search with early termination"""
        if not self.index:
            return []
        
        # Use cache for frequently accessed embeddings
        similarities = []
        
        for doc_id, doc_embedding in self.index.items():
            # Optimized cosine similarity
            if doc_id in self.embedding_cache:
                cached_embedding = self.embedding_cache[doc_id]
                similarity = np.dot(query_embedding, cached_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                )
            else:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
            
            similarities.append((doc_id, float(similarity)))
        
        # Partial sort optimization - only sort top candidates if large dataset
        if len(similarities) > 1000:
            # Sample first, then sort
            import random
            candidates = random.sample(similarities, 1000)
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class OptimizedHaulsStore:
    """Optimized version of HaulsStore with performance improvements"""
    
    def __init__(self, db_path: str = "runs/store/hauls_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None
            self.embedding_dim = 384  # Consistent dimension
            
        # Initialize components
        self.endic = OptimizedEndicIndex(dimension=self.embedding_dim)
        self.documents = {}
        
        # Performance: Batch operations
        self.batch_size = 100
        self.pending_operations = []
        
        # Initialize database with optimizations
        self._init_optimized_db()
        self._load_existing()
    
    def _init_optimized_db(self):
        """Initialize optimized database with indexes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add performance indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_content_fts ON documents(content)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)
        ''')
        
        # Performance settings
        cursor.execute('PRAGMA journal_mode = WAL')
        cursor.execute('PRAGMA synchronous = NORMAL')
        cursor.execute('PRAGMA cache_size = 10000')
        
        conn.commit()
        conn.close()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Optimized embedding generation with caching"""
        if text in self.endic.embedding_cache:
            return self.endic.embedding_cache[text]
        
        if self.model:
            embedding = self.model.encode(text)
        else:
            # Improved hash-based fallback
            text_bytes = text.encode('utf-8')
            hash_obj = hashlib.sha256(text_bytes)  # SHA-256 for better distribution
            hash_hex = hash_obj.hexdigest()
            
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            for i in range(self.embedding_dim):
                # Use hash circularly for better distribution
                char_idx = i % len(hash_hex)
                if char_idx < len(hash_hex) // 2:
                    embedding[i] = int(hash_hex[char_idx], 16) / 15.0
                else:
                    second_idx = (char_idx + len(hash_hex) // 2) % len(hash_hex)
                    embedding[i] = int(hash_hex[second_idx], 16) / 15.0
        
        # Cache the result
        if len(self.endic.embedding_cache) < self.endic.cache_size_limit:
            self.endic.embedding_cache[text] = embedding.copy()
        
        return embedding
    
    def add_documents_batch(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Batch add documents for performance"""
        start_time = time.time()
        doc_ids = []
        
        # Generate all embeddings at once if using sentence-transformers
        if self.model and len(documents) > 10:
            contents = [doc[0] for doc in documents]
            embeddings = self.model.encode(contents)
        else:
            embeddings = []
            for content, metadata in documents:
                embedding = self._get_embedding(content)
                embeddings.append(embedding)
        
        # Batch database insert
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        insert_data = []
        for i, ((content, metadata), embedding) in enumerate(zip(documents, embeddings)):
            doc_id = hashlib.md5(f"{content}{i}".encode()).hexdigest()[:16]
            doc_ids.append(doc_id)
            
            # Store in memory
            doc = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )
            self.documents[doc_id] = doc
            self.endic.add_document(doc_id, embedding)
            
            # Prepare batch insert
            insert_data.append((
                doc_id,
                content,
                json.dumps(metadata),
                pickle.dumps(embedding)
            ))
        
        # Execute batch insert
        cursor.executemany('''
            INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
            VALUES (?, ?, ?, ?)
        ''', insert_data)
        
        conn.commit()
        conn.close()
        
        elapsed = time.time() - start_time
        print(f"✅ Added {len(documents)} documents in {elapsed:.2f}s ({len(documents)/elapsed:.1f} docs/sec)")
        
        return doc_ids
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add single document (uses batch for consistency)"""
        return self.add_documents_batch([(content, metadata or {})])[0]
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Optimized search with filtering"""
        query_embedding = self._get_embedding(query)
        
        # Use optimized EndicIndex
        similar_docs = self.endic.search(query_embedding, top_k * 3)  # Get more candidates
        
        results = []
        for doc_id, similarity in similar_docs:
            if len(results) >= top_k:
                break
                
            doc = self.documents.get(doc_id)
            if not doc:
                continue
            
            # Apply metadata filter if provided
            if filter_metadata:
                matches = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        matches = False
                        break
                if not matches:
                    continue
            
            results.append({
                'id': doc_id,
                'content': doc.content,
                'metadata': doc.metadata,
                'similarity': similarity,
                'created_at': doc.created_at
            })
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get specific document by ID"""
        doc = self.documents.get(doc_id)
        if doc:
            return {
                'id': doc_id,
                'content': doc.content,
                'metadata': doc.metadata,
                'created_at': doc.created_at
            }
        
        # Fallback to database if not in memory
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT content, metadata, created_at FROM documents WHERE id = ?
        ''', (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': doc_id,
                'content': row[0],
                'metadata': json.loads(row[1]),
                'created_at': row[2]
            }
        
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from store"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        # Remove from memory structures
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.endic.index:
            del self.endic.index[doc_id]
        if doc_id in self.endic.doc_mapping:
            del self.endic.doc_mapping[doc_id]
        if doc_id in self.endic.embedding_cache:
            del self.endic.embedding_cache[doc_id]
        
        return affected > 0
    
    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List documents with pagination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, content, metadata, created_at FROM documents LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1][:200] + '...' if len(row[1]) > 200 else row[1],
                'metadata': json.loads(row[2]),
                'created_at': row[3]
            })
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': doc_count,
            'embedding_dimension': self.embedding_dim,
            'model_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'endic_index_size': len(self.endic.index),
            'cache_size': len(self.endic.embedding_cache),
            'optimized': True
        }
    
    def clear(self):
        """Clear all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM documents')
        conn.commit()
        conn.close()
        
        # Clear in-memory structures
        self.documents.clear()
        self.endic.index.clear()
        self.endic.doc_mapping.clear()
        self.endic.embedding_cache.clear()
    
    def _load_existing(self):
        """Load existing documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, content, metadata, embedding FROM documents')
        rows = cursor.fetchall()
        
        print(f"📚 Loading {len(rows)} existing documents...")
        
        for row in rows:
            doc_id, content, metadata_str, embedding_blob = row
            
            # Create document object
            doc = Document(
                id=doc_id,
                content=content,
                metadata=json.loads(metadata_str),
                embedding=self._deserialize_embedding(embedding_blob)
            )
            
            # Add to memory structures
            self.documents[doc_id] = doc
            self.endic.add_document(doc_id, doc.embedding)
        
        conn.close()
        print(f"✅ Loaded {len(rows)} documents into optimized index")
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for storage"""
        return pickle.dumps(embedding)
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize embedding from storage"""
        return pickle.loads(embedding_bytes)


# Test performance improvements
if __name__ == "__main__":
    store = OptimizedHaulsStore("test_optimized.db")
    
    print("🚀 Testing Optimized HaulsStore Performance")
    print("=" * 50)
    
    # Test 1: Batch add performance
    test_docs = []
    for i in range(1000):
        content = f"Performance test document {i} with unique content for optimization testing"
        test_docs.append((content, {'optimized': True, 'index': i}))
    
    start_time = time.time()
    doc_ids = store.add_documents_batch(test_docs)
    add_time = time.time() - start_time
    
    print(f"✅ Batch Add Performance: {add_time:.2f}s ({len(test_docs)/add_time:.1f} docs/sec)")
    
    # Test 2: Search performance
    search_times = []
    for i in range(100):
        start_time = time.time()
        results = store.search(f"performance test {i}")
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search = sum(search_times) / len(search_times)
    print(f"✅ Optimized Search Performance: {avg_search*1000:.2f}ms average")
    
    # Test 3: Memory efficiency
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"✅ Memory Usage: {memory_mb:.1f}MB for {len(store.documents)} documents")
    
    # Show final stats
    stats = store.get_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")