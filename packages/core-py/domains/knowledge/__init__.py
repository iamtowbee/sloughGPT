"""
Knowledge Domain - Core system for managing facts and information.

This module provides a clean API for knowledge management:
- Add knowledge items
- Search knowledge
- Query by category
- Delete knowledge
- Get knowledge for context injection

Usage:
    from domains.knowledge import KnowledgeSystem
    
    kb = KnowledgeSystem()
    kb.add("The sky is blue", category="facts")
    results = kb.search("sky")
    context = kb.get_context()
"""

from .system import KnowledgeSystem, KnowledgeItem

__all__ = ["KnowledgeSystem", "KnowledgeItem"]