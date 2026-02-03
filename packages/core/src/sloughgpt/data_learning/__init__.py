"""Data learning module with autonomous knowledge management."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import uuid
import json
import os
from pathlib import Path


@dataclass
class DataSource:
    id: str
    name: str
    type: str  # "file", "api", "web"
    path: str
    format: str
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeItem:
    id: str
    text: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    quality_score: float
    created_at: datetime


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class DatasetPipeline:
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.knowledge_items: List[KnowledgeItem] = []
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.quality_threshold = 0.85
        self.similarity_threshold = 0.8
    
    def add_source(self, name: str, path: str, format: str, 
                   type: str = "file", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a data source to the pipeline."""
        source_id = str(uuid.uuid4())
        
        data_source = DataSource(
            id=source_id,
            name=name,
            type=type,
            path=path,
            format=format,
            last_updated=datetime.now(),
            metadata=metadata
        )
        
        self.data_sources[source_id] = data_source
        return source_id
    
    async def start_learning(self, config: Dict[str, Any]) -> str:
        """Start autonomous learning from data sources."""
        job_id = str(uuid.uuid4())
        
        # Update configuration
        self.embedding_model = config.get("embedding_model", self.embedding_model)
        self.quality_threshold = config.get("quality_threshold", self.quality_threshold)
        self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
        
        # Process each data source
        for source in self.data_sources.values():
            await self._process_data_source(source)
        
        # Deduplicate knowledge items
        await self._deduplicate_knowledge()
        
        return job_id
    
    async def _process_data_source(self, source: DataSource) -> None:
        """Process a single data source."""
        try:
            if source.type == "file":
                items = await self._process_file_source(source)
            elif source.type == "api":
                items = await self._process_api_source(source)
            elif source.type == "web":
                items = await self._process_web_source(source)
            else:
                items = []
            
            # Filter by quality and add to knowledge base
            for item in items:
                if item.quality_score >= self.quality_threshold:
                    self.knowledge_items.append(item)
        
        except Exception as e:
            print(f"Error processing source {source.name}: {e}")
    
    async def _process_file_source(self, source: DataSource) -> List[KnowledgeItem]:
        """Process file-based data source."""
        items = []
        
        if source.format == "markdown":
            items = await self._process_markdown_files(source.path)
        elif source.format == "json":
            items = await self._process_json_files(source.path)
        elif source.format == "text":
            items = await self._process_text_files(source.path)
        
        return items
    
    async def _process_markdown_files(self, path: str) -> List[KnowledgeItem]:
        """Process markdown files."""
        items = []
        file_path = Path(path)
        
        if file_path.is_file():
            files = [file_path]
        else:
            files = list(file_path.glob("**/*.md"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = self._chunk_text(content)
                
                for chunk in chunks:
                    item = KnowledgeItem(
                        id=str(uuid.uuid4()),
                        text=chunk,
                        embedding=None,  # Will be generated later
                        metadata={"source": str(file_path), "type": "markdown"},
                        quality_score=self._calculate_quality_score(chunk),
                        created_at=datetime.now()
                    )
                    items.append(item)
            
            except Exception as e:
                print(f"Error processing markdown file {file_path}: {e}")
        
        return items
    
    async def _process_json_files(self, path: str) -> List[KnowledgeItem]:
        """Process JSON files."""
        items = []
        file_path = Path(path)
        
        if file_path.is_file():
            files = [file_path]
        else:
            files = list(file_path.glob("**/*.json"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text from JSON
                text_content = self._extract_text_from_json(data)
                
                if text_content:
                    chunks = self._chunk_text(text_content)
                    
                    for chunk in chunks:
                        item = KnowledgeItem(
                            id=str(uuid.uuid4()),
                            text=chunk,
                            embedding=None,
                            metadata={"source": str(file_path), "type": "json"},
                            quality_score=self._calculate_quality_score(chunk),
                            created_at=datetime.now()
                        )
                        items.append(item)
            
            except Exception as e:
                print(f"Error processing JSON file {file_path}: {e}")
        
        return items
    
    async def _process_text_files(self, path: str) -> List[KnowledgeItem]:
        """Process plain text files."""
        items = []
        file_path = Path(path)
        
        if file_path.is_file():
            files = [file_path]
        else:
            files = list(file_path.glob("**/*.txt"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self._chunk_text(content)
                
                for chunk in chunks:
                    item = KnowledgeItem(
                        id=str(uuid.uuid4()),
                        text=chunk,
                        embedding=None,
                        metadata={"source": str(file_path), "type": "text"},
                        quality_score=self._calculate_quality_score(chunk),
                        created_at=datetime.now()
                    )
                    items.append(item)
            
            except Exception as e:
                print(f"Error processing text file {file_path}: {e}")
        
        return items
    
    async def _process_api_source(self, source: DataSource) -> List[KnowledgeItem]:
        """Process API-based data source."""
        # Placeholder for API processing
        return []
    
    async def _process_web_source(self, source: DataSource) -> List[KnowledgeItem]:
        """Process web-based data source."""
        # Placeholder for web scraping
        return []
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def _extract_text_from_json(self, data: Any) -> str:
        """Extract text content from JSON data."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            texts = []
            for value in data.values():
                text = self._extract_text_from_json(value)
                if text:
                    texts.append(text)
            return " ".join(texts)
        elif isinstance(data, list):
            texts = []
            for item in data:
                text = self._extract_text_from_json(item)
                if text:
                    texts.append(text)
            return " ".join(texts)
        else:
            return str(data) if data else ""
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for text content."""
        if not text:
            return 0.0
        
        # Simple quality metrics
        length_score = min(len(text) / 100, 1.0)  # Favor longer texts up to 100 chars
        word_count = len(text.split())
        word_score = min(word_count / 20, 1.0)  # Favor at least 20 words
        
        # Penalty for very short or very long texts
        if len(text) < 50:
            length_score *= 0.5
        elif len(text) > 5000:
            length_score *= 0.8
        
        return (length_score + word_score) / 2
    
    async def _deduplicate_knowledge(self) -> None:
        """Remove duplicate knowledge items based on similarity."""
        # Simple deduplication based on exact text matches
        seen_texts = set()
        unique_items = []
        
        for item in self.knowledge_items:
            if item.text not in seen_texts:
                seen_texts.add(item.text)
                unique_items.append(item)
        
        self.knowledge_items = unique_items
    
    def search_knowledge(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search knowledge base for relevant content."""
        # Simple text-based search (in production, use vector embeddings)
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_items:
            if query_lower in item.text.lower():
                score = len(query.split()) / len(item.text.split())  # Simple relevance score
                
                results.append(SearchResult(
                    id=item.id,
                    text=item.text,
                    score=score,
                    metadata=item.metadata
                ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_items": len(self.knowledge_items),
            "total_sources": len(self.data_sources),
            "avg_quality_score": sum(item.quality_score for item in self.knowledge_items) / len(self.knowledge_items) if self.knowledge_items else 0,
            "embedding_model": self.embedding_model,
            "quality_threshold": self.quality_threshold
        }