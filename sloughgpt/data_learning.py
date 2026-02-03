#!/usr/bin/env python3
"""
SloughGPT Data Learning Pipeline
Automated dataset learning and continuous improvement system
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import aiohttp
import aiofiles
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import hashlib

from .config import SloughGPTConfig, LearningConfig
from .core.database import get_db_session, LearningExperience, KnowledgeNode
from .core.exceptions import create_error, LearningError
from .neural_network import SloughGPT

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    url: Optional[str] = None
    file_path: Optional[str] = None
    source_type: str = "api"  # "api", "file", "web"
    format: str = "json"     # "json", "txt", "csv", "markdown"
    quality_threshold: float = 0.8
    max_size_mb: int = 100
    update_interval: int = 3600  # seconds
    auth_headers: Optional[Dict[str, str]] = None
    preprocessing_steps: List[str] = None
    
    def __post_init__(self):
        if self.preprocessing_steps is None:
            self.preprocessing_steps = ["clean_text", "remove_duplicates", "quality_filter"]

@dataclass
class LearningConfigAdvanced:
    """Advanced learning configuration"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 1000
    similarity_threshold: float = 0.7
    min_chunk_length: int = 50
    max_chunk_length: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 32
    quality_threshold: float = 0.85
    deduplication_threshold: float = 0.95
    update_frequency: int = 3600  # seconds

class DataProcessor:
    """Advanced data processing and chunking"""
    
    def __init__(self, config: LearningConfigAdvanced):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\'\"\/\\]', '', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        return text.strip()
    
    def chunk_document(self, text: str) -> List[str]:
        """Intelligent document chunking"""
        if not text:
            return []
            
        # Clean text first
        text = self.clean_text(text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is short enough, add to current chunk
            if len(current_chunk + paragraph + "\n") <= self.config.max_chunk_length:
                current_chunk += paragraph + "\n"
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= self.config.min_chunk_length:
                    chunks.append(current_chunk.strip())
                
                # Handle long paragraphs
                if len(paragraph) > self.config.max_chunk_length:
                    # Split long paragraph
                    words = paragraph.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + word + " ") <= self.config.max_chunk_length:
                            temp_chunk += word + " "
                        else:
                            if len(temp_chunk) >= self.config.min_chunk_length:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + "\n"
        
        # Add final chunk
        if len(current_chunk) >= self.config.min_chunk_length:
            chunks.append(current_chunk.strip())
            
        # Limit chunks per document
        return chunks[:self.config.max_chunks_per_document]
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute text embedding"""
        return self.embedding_model.encode(text, normalize_embeddings=True)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts"""
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        return np.dot(emb1, emb2)

class VectorStore:
    """FAISS-based vector storage for efficient similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        self.texts = []
        self.metadata = []
        
    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Add texts to vector store"""
        if metadata is None:
            metadata = [{}] * len(texts)
            
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """Search for similar texts"""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx >= 0:
                results.append((self.texts[idx], float(score), self.metadata[idx]))
                
        return results

class DatasetPipeline:
    """Complete dataset learning pipeline"""
    
    def __init__(self, config: Optional[LearningConfigAdvanced] = None):
        self.config = config or LearningConfigAdvanced()
        self.data_sources: List[DataSource] = []
        self.processor = DataProcessor(self.config)
        self.vector_store = VectorStore()
        self.knowledge_graph = {}
        self.learning_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "knowledge_nodes": 0,
            "last_update": None
        }
        
    def add_source(self, name: str, source: str, **kwargs) -> 'DatasetPipeline':
        """Add a data source"""
        if source.startswith('http'):
            data_source = DataSource(name=name, url=source, **kwargs)
        else:
            data_source = DataSource(name=name, file_path=source, source_type="file", **kwargs)
            
        self.data_sources.append(data_source)
        return self
        
    async def fetch_data_from_url(self, source: DataSource) -> str:
        """Fetch data from URL"""
        headers = source.auth_headers or {}
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(source.url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {len(content)} characters from {source.name}")
                        return content
                    else:
                        raise LearningError(f"Failed to fetch data: HTTP {response.status}")
        except Exception as e:
            raise create_error(LearningError, f"Error fetching data from {source.name}: {str(e)}", None, cause=e)
    
    async def load_data_from_file(self, source: DataSource) -> str:
        """Load data from file"""
        try:
            file_path = Path(source.file_path)
            if not file_path.exists():
                raise LearningError(f"File not found: {source.file_path}")
                
            if file_path.stat().st_size > source.max_size_mb * 1024 * 1024:
                raise LearningError(f"File too large: {file_path.stat().st_size} bytes")
                
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            logger.info(f"Successfully loaded {len(content)} characters from {source.name}")
            return content
            
        except Exception as e:
            raise create_error(LearningError, f"Error loading data from {source.name}: {str(e)}", None, cause=e)
    
    def parse_content(self, content: str, format: str) -> List[Dict[str, Any]]:
        """Parse content based on format"""
        if format == "json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                else:
                    return [{"content": str(data)}]
            except json.JSONDecodeError:
                return [{"content": content}]
                
        elif format == "csv":
            import csv
            import io
            reader = csv.DictReader(io.StringIO(content))
            return list(reader)
            
        elif format == "markdown":
            # Split markdown by headers
            sections = []
            lines = content.split('\n')
            current_section = {"title": "", "content": ""}
            
            for line in lines:
                if line.startswith('#'):
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"title": line, "content": ""}
                else:
                    current_section["content"] += line + "\n"
                    
            if current_section["content"]:
                sections.append(current_section)
                
            return sections
            
        else:  # text or unknown
            return [{"content": content}]
    
    def assess_quality(self, text: str) -> float:
        """Assess quality of text content"""
        score = 0.0
        
        # Length score
        length = len(text)
        if self.config.min_chunk_length <= length <= self.config.max_chunk_length:
            score += 0.3
        elif length < self.config.min_chunk_length:
            score -= 0.2
            
        # Vocabulary diversity
        words = text.lower().split()
        if words:
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            score += diversity * 0.3
            
        # Sentence structure
        sentences = text.split('.')
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 25:
                score += 0.2
                
        # Information content (basic heuristic)
        info_keywords = ['because', 'therefore', 'however', 'although', 'since', 'while', 'whereas']
        keyword_count = sum(1 for word in info_keywords if word in text.lower())
        score += min(keyword_count * 0.05, 0.2)
        
        return min(max(score, 0.0), 1.0)
    
    def remove_duplicates(self, chunks: List[str]) -> List[str]:
        """Remove duplicate chunks using semantic similarity"""
        unique_chunks = []
        embeddings = []
        
        for chunk in chunks:
            embedding = self.processor.compute_embedding(chunk)
            
            # Check against existing chunks
            is_duplicate = False
            for existing_embedding in embeddings:
                similarity = np.dot(embedding, existing_embedding)
                if similarity > self.config.deduplication_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_chunks.append(chunk)
                embeddings.append(embedding)
                
        return unique_chunks
    
    async def process_source(self, source: DataSource) -> Dict[str, Any]:
        """Process a single data source"""
        logger.info(f"Processing data source: {source.name}")
        
        # Fetch data
        if source.source_type == "api" and source.url:
            content = await self.fetch_data_from_url(source)
        elif source.source_type == "file" and source.file_path:
            content = await self.load_data_from_file(source)
        else:
            raise LearningError(f"Invalid source configuration for {source.name}")
            
        # Parse content
        documents = self.parse_content(content, source.format)
        
        # Process documents
        all_chunks = []
        processed_docs = 0
        
        for doc in documents:
            # Extract text content
            if isinstance(doc, dict):
                text = doc.get('content', str(doc))
                doc_metadata = {k: v for k, v in doc.items() if k != 'content'}
            else:
                text = str(doc)
                doc_metadata = {}
                
            # Create chunks
            chunks = self.processor.chunk_document(text)
            
            # Apply preprocessing steps
            if "remove_duplicates" in source.preprocessing_steps:
                chunks = self.remove_duplicates(chunks)
                
            if "quality_filter" in source.preprocessing_steps:
                quality_chunks = []
                for chunk in chunks:
                    quality = self.assess_quality(chunk)
                    if quality >= source.quality_threshold:
                        all_chunks.append({
                            "text": chunk,
                            "metadata": {
                                **doc_metadata,
                                "source": source.name,
                                "quality": quality,
                                "processed_at": datetime.utcnow().isoformat()
                            }
                        })
                        
            else:
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **doc_metadata,
                            "source": source.name,
                            "processed_at": datetime.utcnow().isoformat()
                        }
                    })
                    
            processed_docs += 1
            
        # Compute embeddings and add to vector store
        if all_chunks:
            texts = [chunk["text"] for chunk in all_chunks]
            metadata = [chunk["metadata"] for chunk in all_chunks]
            embeddings = self.processor.embedding_model.encode(texts, normalize_embeddings=True)
            
            self.vector_store.add_texts(texts, embeddings, metadata)
            
            # Store in database
            await self.store_knowledge(all_chunks, embeddings)
            
        return {
            "source": source.name,
            "documents_processed": processed_docs,
            "chunks_created": len(all_chunks),
            "quality_score": np.mean([chunk["metadata"].get("quality", 0.5) for chunk in all_chunks])
        }
    
    async def store_knowledge(self, chunks: List[Dict], embeddings: np.ndarray):
        """Store knowledge in database"""
        try:
            with get_db_session() as session:
                for i, chunk in enumerate(chunks):
                    # Create knowledge node
                    node = KnowledgeNode(
                        node_id=f"{chunk['metadata']['source']}_{hash(chunk['text']) % 1000000}",
                        content=chunk["text"],
                        node_type="text_chunk",
                        importance=chunk["metadata"].get("quality", 0.5),
                        embedding=embeddings[i].tolist(),
                        metadata_json=chunk["metadata"]
                    )
                    session.add(node)
                    
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            
    async def start_learning(self, config: Optional[Dict] = None) -> str:
        """Start the learning pipeline"""
        if config:
            # Update config
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        logger.info("Starting SloughGPT learning pipeline")
        
        # Process all data sources
        results = []
        for source in self.data_sources:
            try:
                result = await self.process_source(source)
                results.append(result)
                logger.info(f"Processed {source.name}: {result}")
            except Exception as e:
                logger.error(f"Failed to process {source.name}: {str(e)}")
                results.append({
                    "source": source.name,
                    "error": str(e)
                })
                
        # Update statistics
        self.learning_stats.update({
            "documents_processed": sum(r.get("documents_processed", 0) for r in results),
            "chunks_created": sum(r.get("chunks_created", 0) for r in results),
            "knowledge_nodes": self.vector_store.index.ntotal,
            "last_update": datetime.utcnow().isoformat()
        })
        
        # Generate job ID
        job_id = hashlib.md5(f"learning_{time.time()}".encode()).hexdigest()
        
        logger.info(f"Learning job {job_id} completed successfully")
        logger.info(f"Total stats: {self.learning_stats}")
        
        return job_id
        
    def search_knowledge(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search through learned knowledge"""
        query_embedding = self.processor.compute_embedding(query)
        results = self.vector_store.search(query_embedding, k=k, threshold=self.config.similarity_threshold)
        
        return [
            {
                "text": text,
                "score": score,
                "metadata": metadata
            }
            for text, score, metadata in results
        ]
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            **self.learning_stats,
            "vector_store_size": self.vector_store.index.ntotal,
            "data_sources": len(self.data_sources)
        }

# Convenience functions
async def create_learning_pipeline() -> DatasetPipeline:
    """Create and return a learning pipeline"""
    return DatasetPipeline()

async def start_autonomous_learning(config: Optional[Dict] = None) -> str:
    """Start autonomous learning with default data sources"""
    pipeline = await create_learning_pipeline()
    
    # Add default data sources
    pipeline.add_source("documentation", "./docs/", format="markdown")
    pipeline.add_source("training_data", "./data/training/", format="json")
    
    return await pipeline.start_learning(config)

# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SloughGPT Data Learning Pipeline")
    parser.add_argument("--add-source", nargs=2, metavar=("NAME", "PATH/URL"), 
                       help="Add a data source")
    parser.add_argument("--format", default="json", choices=["json", "txt", "csv", "markdown"],
                       help="Source format")
    parser.add_argument("--start-learning", action="store_true",
                       help="Start the learning pipeline")
    parser.add_argument("--search", metavar="QUERY",
                       help="Search through learned knowledge")
    parser.add_argument("--stats", action="store_true",
                       help="Show learning statistics")
    parser.add_argument("--config", help="Configuration file")
    
    async def main():
        args = parser.parse_args()
        
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = {}
            
        pipeline = DatasetPipeline()
        
        if args.add_source:
            name, source = args.add_source
            pipeline.add_source(name, source, format=args.format)
            print(f"✅ Added data source: {name} -> {source}")
            
        if args.start_learning:
            job_id = await pipeline.start_learning(config)
            print(f"🚀 Learning started successfully! Job ID: {job_id}")
            
        if args.search:
            results = pipeline.search_knowledge(args.search)
            print(f"\n🔍 Search results for '{args.search}':")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.3f}")
                print(f"   Text: {result['text'][:200]}...")
                
        if args.stats:
            stats = pipeline.get_learning_stats()
            print("\n📊 Learning Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
    
    asyncio.run(main())