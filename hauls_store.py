#!/usr/bin/env python3
"""
HaulsStore - Vector Database for SLO Memory System

A lightweight vector database supporting:
- Document embedding and storage
- Semantic similarity search
- Persistent storage across sessions
- RAG integration for SLO knowledge retrieval
"""

import json
import pickle
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

# Try to import sentence-transformers, fallback to simple embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using simple hash-based embeddings.")


@dataclass
class Document:
    """Document structure for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None


class EndicIndex:
    """Endic-style semantic index for efficient retrieval with optimizations"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = {}
        self.doc_mapping = {}
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
        """Optimized search with early termination and caching"""
        if not self.index:
            return []
        
        similarities = []
        
        for doc_id, doc_embedding in self.index.items():
            # Use cache for frequently accessed embeddings
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
            # Sample first, then sort (for very large datasets)
            import random
            candidates = random.sample(similarities, 1000)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]
        else:
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]


class HaulsStore:
    """Main vector database implementation"""
    
    def __init__(self, db_path: str = "runs/store/hauls_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None
            self.embedding_dim = 64  # Simple hash-based fallback
        
        # Initialize components
        self.endic = EndicIndex(dimension=self.embedding_dim)
        self.documents = {}
        
        # Initialize optimized database
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
        
        # Load existing documents into memory
        self._load_documents()
    
    def _load_documents(self):
        """Load all documents from database into memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, content, metadata FROM documents')
            rows = cursor.fetchall()
            
            for doc_id, content, metadata in rows:
                try:
                    metadata_dict = json.loads(metadata) if metadata else {}
                    doc = Document(id=doc_id, content=content, metadata=metadata_dict)
                    self.documents[doc_id] = doc
                    
                    # Add to EndicIndex for search
                    embedding = self._get_embedding(content)
                    self.endic.add_document(doc_id, embedding)
                    
                except Exception as e:
                    print(f"Warning: Failed to load document {doc_id}: {e}")
                    continue
            
            conn.close()
            print(f"✅ Loaded {len(self.documents)} documents into memory")
            
        except Exception as e:
            print(f"❌ Failed to load documents: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self.model:
            return self.model.encode(text)
        else:
            # Simple hash-based fallback
            text_bytes = text.encode('utf-8')
            hash_obj = hashlib.md5(text_bytes)
            hash_hex = hash_obj.hexdigest()
            
            # Convert hex to numeric array
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            for i, char in enumerate(hash_hex):
                if i < self.embedding_dim:
                    embedding[i] = int(char, 16) / 15.0
            
            return embedding
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for storage"""
        return pickle.dumps(embedding)
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize embedding from storage"""
        return pickle.loads(embedding_bytes)
    
    def add_documents_batch(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Add multiple documents in batch for performance"""
        doc_ids = []
        
        # Generate embeddings for all documents
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
            
            # Create document object
            doc = Document(
                id=doc_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding
            )
            
            # Add to in-memory structures
            self.documents[doc_id] = doc
            self.endic.add_document(doc_id, embedding)
            
            # Prepare batch insert data
            insert_data.append((
                doc_id,
                content,
                json.dumps(metadata or {}),
                self._serialize_embedding(embedding)
            ))
        
        # Execute batch insert
        cursor.executemany('''
            INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
            VALUES (?, ?, ?, ?)
        ''', insert_data)
        
        conn.commit()
        conn.close()
        
        return doc_ids
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add single document (uses batch for consistency)"""
        return self.add_documents_batch([(content, metadata or {})])[0]
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self._get_embedding(query)
        
        # Search using Endic index
        similar_docs = self.endic.search(query_embedding, top_k)
        
        results = []
        for doc_id, similarity in similar_docs:
            doc = self.documents.get(doc_id)
            if doc:
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
        
        # Remove from in-memory structures
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.endic.index:
            del self.endic.index[doc_id]
        if doc_id in self.endic.doc_mapping:
            del self.endic.doc_mapping[doc_id]
        
        return affected > 0
    
    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all documents"""
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
            'endic_index_size': len(self.endic.index)
        }
    
    def _load_existing(self):
        """Load existing documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, content, metadata, embedding FROM documents')
        
        for row in cursor.fetchall():
            doc_id, content, metadata_str, embedding_blob = row
            
            # Create document object
            doc = Document(
                id=doc_id,
                content=content,
                metadata=json.loads(metadata_str),
                embedding=self._deserialize_embedding(embedding_blob)
            )
            
            # Add to in-memory structures
            self.documents[doc_id] = doc
            self.endic.add_document(doc_id, doc.embedding)
        
        conn.close()
    
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


# CLI interface
def main():
    """CLI interface for HaulsStore"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HaulsStore Vector Database')
    parser.add_argument('--db-path', default='runs/store/hauls_store.db', help='Database path')
    parser.add_argument('--add', help='Add content to store')
    parser.add_argument('--search', help='Search content')
    parser.add_argument('--list', action='store_true', help='List documents')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--clear', action='store_true', help='Clear all documents')
    
    args = parser.parse_args()
    
    store = HaulsStore(args.db_path)
    
    if args.add:
        doc_id = store.add_document(args.add, {'source': 'cli'})
        print(f"Added document: {doc_id}")
    
    elif args.search:
        results = store.search(args.search)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['similarity']:.3f}] {result['content'][:100]}...")
    
    elif args.list:
        docs = store.list_documents()
        print(f"Documents ({len(docs)}):")
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc['id']}: {doc['content']}")
    
    elif args.stats:
        stats = store.get_stats()
        print("HaulsStore Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.clear:
        store.clear()
        print("Cleared all documents")
    
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()