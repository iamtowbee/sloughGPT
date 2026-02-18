"""
RAG System - Ported from recovered slo_rag.py
Retrieval-Augmented Generation for knowledge retrieval
"""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path


class RAGSystem:
    """RAG system for knowledge retrieval"""
    
    def __init__(self, store_path: str = "runs/store/rag_store.db"):
        self.store_path = store_path
        self.documents: List[Dict] = []
        self.max_context_length = 2000
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a document to the knowledge base."""
        doc = {
            "content": content,
            "metadata": metadata or {},
            "id": len(self.documents)
        }
        self.documents.append(doc)
    
    def add_training_knowledge(self, dataset_path: str, metadata: Optional[Dict] = None) -> int:
        """Add dataset information to knowledge base."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return 0
        
        added = 0
        if dataset_path.suffix == '.jsonl':
            with open(dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            content = str(data.get('text', data))
                            if content:
                                doc_metadata = {
                                    **(metadata or {}),
                                    'source': 'training_dataset',
                                    'dataset_name': dataset_path.name,
                                    'line_number': i
                                }
                                self.add_document(content, doc_metadata)
                                added += 1
                        except json.JSONDecodeError:
                            continue
        
        return added
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            if score > 0:
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_context(self, query: str, max_length: Optional[int] = None) -> str:
        """Get context from retrieved documents."""
        max_length = max_length or self.max_context_length
        results = self.search(query, top_k=10)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result["content"]
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear all documents."""
        self.documents = []
    
    def count(self) -> int:
        """Count total documents."""
        return len(self.documents)


class VectorStore:
    """Simple vector store for embeddings (placeholder)."""
    
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add(self, key: str, vector: List[float], metadata: Optional[Dict] = None) -> None:
        """Add a vector."""
        self.vectors[key] = vector
        self.metadata[key] = metadata or {}
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar vectors (simple cosine similarity)."""
        results = []
        
        for key, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            results.append({
                "key": key,
                "score": similarity,
                "metadata": self.metadata.get(key, {})
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


__all__ = ["RAGSystem", "VectorStore"]
