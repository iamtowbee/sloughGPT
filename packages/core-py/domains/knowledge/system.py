"""
Knowledge System - Core domain for managing facts and context injection.

This system provides:
- Storage of knowledge items
- Search functionality
- Category-based organization
- Usage tracking
- Context formatting for AI injection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger("knowledge")


@dataclass
class KnowledgeItem:
    """A single piece of knowledge/fact."""
    id: str
    content: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    usage_count: int = 0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeItem":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            usage_count=data.get("usage_count", 0),
        )


class KnowledgeSystem:
    """
    Core knowledge system for managing facts and context.
    
    Usage:
        kb = KnowledgeSystem()
        kb.add("The sky is blue")
        kb.add("Water is essential for life", category="biology")
        kb.search("sky")
        kb.get_context()  # Returns formatted context for AI injection
    """
    
    def __init__(self, storage_path: str = "data/knowledge.json"):
        self.storage_path = storage_path
        self.items: List[KnowledgeItem] = []
        self._counter = 0
        self._load()
    
    def _load(self):
        """Load knowledge from storage."""
        import os
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.items = [KnowledgeItem.from_dict(d) for d in data.get("items", [])]
                    self._counter = data.get("counter", 0)
                logger.info(f"Loaded {len(self.items)} knowledge items")
            except Exception as e:
                logger.warning(f"Failed to load knowledge: {e}")
    
    def _save(self):
        """Save knowledge to storage."""
        import os
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "items": [item.to_dict() for item in self.items],
                    "counter": self._counter,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save knowledge: {e}")
    
    def add(self, content: str, category: str = "general", tags: Optional[List[str]] = None) -> KnowledgeItem:
        """
        Add a new knowledge item.
        
        Args:
            content: The fact or information to store
            category: Category for organization (default: "general")
            tags: Optional list of tags
            
        Returns:
            The created KnowledgeItem
        """
        if not content.strip():
            raise ValueError("Content cannot be empty")
        
        item = KnowledgeItem(
            id=f"know_{self._counter}",
            content=content.strip(),
            category=category,
            tags=tags or [],
        )
        self.items.append(item)
        self._counter += 1
        self._save()
        
        logger.info(f"Added knowledge item: {item.id}")
        return item
    
    def get(self, id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        for item in self.items:
            if item.id == id:
                item.usage_count += 1
                self._save()
                return item
        return None
    
    def search(self, query: str, category: Optional[str] = None) -> List[KnowledgeItem]:
        """
        Search knowledge items.
        
        Args:
            query: Search term
            category: Optional category filter
            
        Returns:
            List of matching items
        """
        results = []
        query_lower = query.lower()
        
        for item in self.items:
            if query_lower in item.content.lower():
                if category is None or item.category == category:
                    results.append(item)
        
        return results
    
    def get_by_category(self, category: str) -> List[KnowledgeItem]:
        """Get all items in a category."""
        return [item for item in self.items if item.category == category]
    
    def delete(self, id: str) -> bool:
        """Delete a knowledge item by ID."""
        original_len = len(self.items)
        self.items = [item for item in self.items if item.id != id]
        
        if len(self.items) < original_len:
            self._save()
            logger.info(f"Deleted knowledge item: {id}")
            return True
        return False
    
    def update(self, id: str, content: Optional[str] = None, category: Optional[str] = None) -> Optional[KnowledgeItem]:
        """Update a knowledge item."""
        item = self.get(id)
        if not item:
            return None
        
        if content:
            item.content = content.strip()
        if category:
            item.category = category
        
        self._save()
        return item
    
    def get_context(self, max_items: int = 10) -> str:
        """
        Get formatted context for AI injection.
        
        Args:
            max_items: Maximum number of items to include
            
        Returns:
            Formatted string for injection into prompts
        """
        if not self.items:
            return ""
        
        sorted_items = sorted(self.items, key=lambda x: x.usage_count, reverse=True)[:max_items]
        context = "[IMPORTANT KNOWLEDGE - Use this information when responding:]\n"
        context += "\n".join([f"• {item.content}" for item in sorted_items])
        context += "\n[/IMPORTANT KNOWLEDGE]"
        
        return context
    
    def get_all(self) -> List[KnowledgeItem]:
        """Get all knowledge items."""
        return self.items
    
    def stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        categories = {}
        for item in self.items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        return {
            "total": len(self.items),
            "categories": len(categories),
            "category_counts": categories,
            "total_usage": sum(item.usage_count for item in self.items),
        }
    
    def clear(self):
        """Clear all knowledge items."""
        self.items = []
        self._save()
        logger.info("Cleared all knowledge items")


# Global instance for easy access
_default_kb: Optional[KnowledgeSystem] = None


def get_knowledge_system() -> KnowledgeSystem:
    """Get the default knowledge system instance."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeSystem()
    return _default_kb