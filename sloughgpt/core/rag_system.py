"""
SloughGPT Knowledge Base & RAG System
Retrieval-Augmented Generation with vector databases and knowledge management
"""

import asyncio
import json
import time
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from sloughgpt.core.logging_system import get_logger, timer
from sloughgpt.core.performance import get_performance_optimizer

class KnowledgeSourceType(Enum):
    """Types of knowledge sources"""
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    API = "api"
    DATABASE = "database"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    CODE_REPOSITORY = "code_repository"
    CONVERSATION_HISTORY = "conversation_history"
    EXTERNAL_KNOWLEDGE = "external_knowledge"

class RetrievalStrategy(Enum):
    """Retrieval strategies for knowledge retrieval"""
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID_SEARCH = "hybrid_search"
    KEYWORD_SEARCH = "keyword_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    RECENT_DOCUMENTS = "recent_documents"
    USER_PREFERENCE = "user_preference"

@dataclass
class KnowledgeDocument:
    """Represents a knowledge document"""
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: KnowledgeSourceType = KnowledgeSourceType.DOCUMENT
    source_url: Optional[str] = None
    author: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None
    parent_doc_id: Optional[str] = None
    access_level: str = "public"
    confidence_score: float = 1.0
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "chunk_id": self.chunk_id,
            "parent_doc_id": self.parent_doc_id,
            "access_level": self.access_level,
            "confidence_score": self.confidence_score,
            "relevance_score": self.relevance_score
        }

@dataclass
class RetrievalResult:
    """Result of knowledge retrieval"""
    query: str
    documents: List[KnowledgeDocument] = field(default_factory=list)
    total_found: int = 0
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH
    search_time: float = 0.0
    relevance_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def top_k_documents(self, k: int = 5) -> List[KnowledgeDocument]:
        """Get top k documents by relevance score"""
        sorted_docs = sorted(self.documents, key=lambda d: d.relevance_score, reverse=True)
        return sorted_docs[:k]

@dataclass
class RAGConfig:
    """RAG system configuration"""
    vector_db_config: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_context_length: int = 4000
    max_retrieved_docs: int = 10
    similarity_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_reranking: bool = True
    enable_hyde: bool = True  # Hypothetical Document Embeddings
    enable_query_expansion: bool = True

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"vector_db_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def add_document(self, doc: KnowledgeDocument) -> str:
        """Add document to vector database"""
        pass
    
    @abstractmethod
    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector database"""
        pass
    
    @abstractmethod
    async def update_document(self, doc: KnowledgeDocument) -> bool:
        """Update document in vector database"""
        pass

class SimpleVectorDB(VectorDatabase):
    """Simple in-memory vector database implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._dimension = 384  # Default embedding dimension
    
    async def add_document(self, doc: KnowledgeDocument) -> str:
        """Add document to vector database"""
        self.logger.debug(f"Adding document {doc.doc_id} to vector DB")
        
        # Generate mock embedding (in real implementation, use embedding model)
        await asyncio.sleep(0.01)  # Simulate embedding generation
        embedding = np.random.rand(self._dimension).astype(np.float32)
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        doc.embeddings = embedding
        self.documents[doc.doc_id] = doc
        self.embeddings[doc.doc_id] = embedding
        
        self.logger.debug(f"Document {doc.doc_id} added successfully")
        return doc.doc_id
    
    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        self.logger.debug(f"Searching vector DB with top_k={top_k}")
        
        await asyncio.sleep(0.005)  # Simulate search time
        
        if not self.embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_vector, doc_embedding)
            similarities.append((doc_id, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector database"""
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        
        self.logger.debug(f"Document {doc_id} deleted")
        return True
    
    async def update_document(self, doc: KnowledgeDocument) -> bool:
        """Update document in vector database"""
        # Delete old embedding
        await self.delete_document(doc.doc_id)
        
        # Add new document
        return await self.add_document(doc)

class EmbeddingGenerator:
    """Generates embeddings for text"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_logger("embedding_generator")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        self.logger.debug(f"Generating embedding for text: {text[:50]}...")
        
        # Mock embedding generation
        await asyncio.sleep(0.01)  # Simulate model inference
        
        # Generate consistent embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        
        embedding = np.random.rand(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

class DocumentChunker:
    """Chunks documents for better retrieval"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = get_logger("document_chunker")
    
    def chunk_document(self, doc: KnowledgeDocument) -> List[KnowledgeDocument]:
        """Chunk document into smaller pieces"""
        self.logger.debug(f"Chunking document {doc.doc_id}")
        
        if len(doc.content) <= self.chunk_size:
            return [doc]
        
        chunks = []
        words = doc.content.split()
        
        i = 0
        chunk_id = 0
        while i < len(words):
            # Calculate chunk boundaries
            start = i - self.overlap if i > 0 else 0
            end = min(i + self.chunk_size, len(words))
            
            if start < 0:
                start = 0
            
            # Create chunk
            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)
            
            chunk = KnowledgeDocument(
                doc_id=f"{doc.doc_id}_chunk_{chunk_id}",
                title=f"{doc.title} (Chunk {chunk_id + 1})",
                content=chunk_content,
                metadata={
                    **doc.metadata,
                    "chunk_index": chunk_id,
                    "word_start": start,
                    "word_end": end,
                    "original_doc_id": doc.doc_id
                },
                source_type=doc.source_type,
                source_url=doc.source_url,
                author=doc.author,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                tags=doc.tags,
                chunk_id=str(chunk_id),
                parent_doc_id=doc.doc_id,
                access_level=doc.access_level,
                confidence_score=doc.confidence_score
            )
            
            chunks.append(chunk)
            
            i = end
            chunk_id += 1
        
        self.logger.debug(f"Document chunked into {len(chunks)} chunks")
        return chunks

class KnowledgeRetriever:
    """Retrieves knowledge from various sources"""
    
    def __init__(self, vector_db: VectorDatabase, config: RAGConfig):
        self.vector_db = vector_db
        self.config = config
        self.embedding_generator = EmbeddingGenerator(config.embedding_model)
        self.chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)
        self.logger = get_logger("knowledge_retriever")
    
    async def add_knowledge(self, doc: KnowledgeDocument) -> str:
        """Add knowledge document with chunking and embedding"""
        self.logger.info(f"Adding knowledge document: {doc.title}")
        
        # Chunk document
        chunks = self.chunker.chunk_document(doc)
        
        # Add chunks to vector database
        chunk_ids = []
        for chunk in chunks:
            # Generate embedding
            chunk.embeddings = await self.embedding_generator.generate_embedding(chunk.content)
            
            # Add to vector database
            chunk_id = await self.vector_db.add_document(chunk)
            chunk_ids.append(chunk_id)
        
        self.logger.info(f"Added {len(chunks)} chunks for document {doc.doc_id}")
        return doc.doc_id
    
    async def retrieve_knowledge(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH,
                              top_k: int = None) -> RetrievalResult:
        """Retrieve knowledge based on query"""
        start_time = time.time()
        
        top_k = top_k or self.config.max_retrieved_docs
        self.logger.info(f"Retrieving knowledge for query: {query[:100]} with strategy: {strategy.value}")
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Search vector database
            search_results = await self.vector_db.search(query_embedding, top_k)
            
            # Retrieve documents
            documents = []
            for doc_id, similarity_score in search_results:
                # Get document from vector database
                # In real implementation, this would query the actual document store
                doc = await self._get_document_by_id(doc_id)
                if doc:
                    doc.relevance_score = similarity_score
                    documents.append(doc)
            
            # Filter by similarity threshold
            filtered_docs = [doc for doc in documents if doc.relevance_score >= self.config.similarity_threshold]
            
            # Apply retrieval strategy
            if strategy == RetrievalStrategy.SEMANTIC_SEARCH:
                final_docs = filtered_docs
            elif strategy == RetrievalStrategy.HYBRID_SEARCH:
                final_docs = await self._hybrid_search(query, filtered_docs)
            elif strategy == RetrievalStrategy.KEYWORD_SEARCH:
                final_docs = await self._keyword_search(query, filtered_docs)
            elif strategy == RetrievalStrategy.GRAPH_TRAVERSAL:
                final_docs = await self._graph_traversal_search(query, filtered_docs)
            else:
                final_docs = filtered_docs
            
            # Sort by relevance
            final_docs.sort(key=lambda d: d.relevance_score, reverse=True)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                query=query,
                documents=final_docs[:top_k],
                total_found=len(final_docs),
                retrieval_strategy=strategy,
                search_time=retrieval_time,
                relevance_scores=[doc.relevance_score for doc in final_docs[:top_k]]
            )
            
            self.logger.info(f"Retrieved {len(final_docs)} documents in {retrieval_time:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Knowledge retrieval failed: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                retrieval_strategy=strategy,
                search_time=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    async def _get_document_by_id(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Get document by ID (mock implementation)"""
        # Mock document retrieval - in real implementation, query document store
        await asyncio.sleep(0.001)
        
        # Create mock document
        doc = KnowledgeDocument(
            doc_id=doc_id,
            title=f"Document {doc_id}",
            content=f"This is the content of document {doc_id}. " * 50,
            source_type=KnowledgeSourceType.DOCUMENT
        )
        
        return doc
    
    async def _hybrid_search(self, query: str, documents: List[KnowledgeDocument]) -> List[KnowledgeDocument]:
        """Hybrid search combining semantic and keyword matching"""
        # Simple hybrid scoring
        query_words = set(query.lower().split())
        
        hybrid_docs = []
        for doc in documents:
            # Combine semantic and keyword scores
            semantic_score = doc.relevance_score
            
            # Calculate keyword match score
            doc_words = set(doc.content.lower().split())
            keyword_matches = len(query_words.intersection(doc_words))
            keyword_score = keyword_matches / len(query_words) if query_words else 0
            
            # Hybrid score (weighted average)
            hybrid_score = 0.7 * semantic_score + 0.3 * keyword_score
            doc.relevance_score = hybrid_score
            
            hybrid_docs.append(doc)
        
        return hybrid_docs
    
    async def _keyword_search(self, query: str, documents: List[KnowledgeDocument]) -> List[KnowledgeDocument]:
        """Keyword-based search"""
        query_words = set(query.lower().split())
        
        keyword_docs = []
        for doc in documents:
            doc_words = set(doc.content.lower().split())
            
            # Calculate keyword relevance
            matches = len(query_words.intersection(doc_words))
            relevance = matches / len(query_words) if query_words else 0
            
            doc.relevance_score = relevance
            keyword_docs.append(doc)
        
        return keyword_docs
    
    async def _graph_traversal_search(self, query: str, documents: List[KnowledgeDocument]) -> List[KnowledgeDocument]:
        """Graph traversal-based search (mock implementation)"""
        # Mock graph traversal - in real implementation, traverse knowledge graph
        return documents[:10]  # Return top 10 for now

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = get_logger("rag_system")
        self.optimizer = get_performance_optimizer()
        
        # Initialize components
        self.vector_db = SimpleVectorDB(config.vector_db_config)
        self.retriever = KnowledgeRetriever(self.vector_db, config)
        
        # Knowledge base
        self.knowledge_base: Dict[str, KnowledgeDocument] = {}
        
        # Performance tracking
        self._performance_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_documents_retrieved": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def ingest_document(self, title: str, content: str, **kwargs) -> str:
        """Ingest a document into the knowledge base"""
        self.logger.info(f"Ingesting document: {title}")
        
        doc_id = str(uuid.uuid4())
        
        doc = KnowledgeDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            metadata=kwargs,
            source_type=KnowledgeSourceType.DOCUMENT
        )
        
        # Add to retriever
        await self.retriever.add_knowledge(doc)
        
        # Store in knowledge base
        self.knowledge_base[doc_id] = doc
        
        self.logger.info(f"Document {doc_id} ingested successfully")
        return doc_id
    
    async def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None,
                          retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH) -> str:
        """Generate response with RAG"""
        self.logger.info(f"Generating RAG response for: {query[:100]}")
        
        with timer("rag_generation"):
            try:
                # Retrieve relevant knowledge
                retrieval_result = await self.retriever.retrieve_knowledge(
                    query, strategy=retrieval_strategy
                )
                
                # Update performance stats
                self._update_performance_stats(retrieval_result)
                
                if not retrieval_result.documents:
                    # No relevant documents found, generate response without context
                    return await self._generate_fallback_response(query)
                
                # Format context
                context_text = self._format_context(retrieval_result)
                
                # Generate response using context
                response = await self._generate_response_with_context(query, context_text, context or {})
                
                # Log RAG metadata
                self.logger.info(f"RAG response generated with {len(retrieval_result.documents)} context documents")
                
                return response
                
            except Exception as e:
                self.logger.error(f"RAG generation failed: {e}")
                return await self._generate_fallback_response(query)
    
    async def _format_context(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved documents as context"""
        if not retrieval_result.documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieval_result.documents):
            context_parts.append(f"Document {i+1}: {doc.title}")
            context_parts.append(f"Content: {doc.content[:500]}...")
            if doc.source_url:
                context_parts.append(f"Source: {doc.source_url}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    async def _generate_response_with_context(self, query: str, context: str, 
                                        user_context: Dict[str, Any]) -> str:
        """Generate response using retrieved context"""
        # Mock LLM call with context
        await asyncio.sleep(0.1)  # Simulate LLM inference
        
        # Create enhanced prompt
        enhanced_prompt = f"""
        Using the following context, answer the user's question:
        
        Context:
        {context}
        
        User Question: {query}
        
        Additional Context: {json.dumps(user_context, indent=2)}
        
        Provide a helpful and accurate answer based on the provided context.
        """
        
        # Mock response generation
        response = f"Based on the retrieved documents, here's what I found about '{query}': "
        response += f"I found {len(context.split('---')) // 3} relevant documents. "
        response += f"According to the context, {self._generate_mock_answer(query)}."
        
        return response
    
    async def _generate_fallback_response(self, query: str) -> str:
        """Generate response without retrieved context"""
        self.logger.info("Generating fallback response without context")
        
        await asyncio.sleep(0.05)  # Faster response without context
        
        return f"I don't have specific information about '{query}' in my knowledge base. "
               f"Could you provide more context or try rephrasing your question?"
    
    def _generate_mock_answer(self, query: str) -> str:
        """Generate mock answer based on query"""
        query_lower = query.lower()
        
        if "how" in query_lower or "what" in query_lower:
            return "here are the key points to consider"
        elif "why" in query_lower:
            return "there are several important factors"
        elif "when" in query_lower:
            return "it typically happens under certain conditions"
        else:
            return "this information should be helpful for your needs"
    
    def _update_performance_stats(self, retrieval_result: RetrievalResult) -> None:
        """Update performance statistics"""
        self._performance_stats["total_queries"] += 1
        
        # Update averages
        total_queries = self._performance_stats["total_queries"]
        current_avg_time = self._performance_stats["avg_retrieval_time"]
        current_avg_docs = self._performance_stats["avg_documents_retrieved"]
        
        self._performance_stats["avg_retrieval_time"] = (
            (current_avg_time * (total_queries - 1) + retrieval_result.search_time) / total_queries
        )
        self._performance_stats["avg_documents_retrieved"] = (
            (current_avg_docs * (total_queries - 1) + len(retrieval_result.documents)) / total_queries
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get RAG system performance statistics"""
        return self._performance_stats.copy()
    
    async def search_knowledge(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH,
                         top_k: int = 10) -> RetrievalResult:
        """Search knowledge base without generation"""
        return await self.retriever.retrieve_knowledge(query, strategy, top_k)
    
    async def delete_knowledge(self, doc_id: str) -> bool:
        """Delete knowledge from the system"""
        success = await self.vector_db.delete_document(doc_id)
        if success and doc_id in self.knowledge_base:
            del self.knowledge_base[doc_id]
        
        return success
    
    async def update_knowledge(self, doc_id: str, **updates) -> bool:
        """Update knowledge in the system"""
        if doc_id not in self.knowledge_base:
            return False
        
        doc = self.knowledge_base[doc_id]
        
        # Update document fields
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        
        doc.updated_at = time.time()
        
        return await self.vector_db.update_document(doc)

# Global RAG system instance
_global_rag_system: Optional[RAGSystem] = None

def get_rag_system(config: Optional[RAGConfig] = None) -> RAGSystem:
    """Get or create global RAG system"""
    global _global_rag_system
    if _global_rag_system is None:
        _global_rag_system = RAGSystem(config or RAGConfig())
    return _global_rag_system

# Decorators for easy use
def rag_enhanced(strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH):
    """Decorator for RAG-enhanced generation"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            rag_system = get_rag_system()
            
            query = kwargs.get("query", str(args[0]) if args else "")
            context = kwargs.get("context", {})
            
            response = await rag_system.generate_response(
                query, context, strategy
            )
            
            return response
        
        return wrapper
    return decorator

def knowledge_retrieval(strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH):
    """Decorator for knowledge retrieval only"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            rag_system = get_rag_system()
            
            query = kwargs.get("query", str(args[0]) if args else "")
            top_k = kwargs.get("top_k", 10)
            
            result = await rag_system.search_knowledge(query, strategy, top_k)
            
            return result
        
        return wrapper
    return decorator