#!/usr/bin/env python3
"""
RAG Integration for SLO

Retrieval-Augmented Generation system that integrates HaulsStore
with SLO's training and inference capabilities.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from hauls_store import HaulsStore
    HAULS_STORE_AVAILABLE = True
except ImportError:
    HAULS_STORE_AVAILABLE = False


class SLO_RAG:
    """RAG system for SLO knowledge retrieval"""
    
    def __init__(self, store_path: str = "runs/store/hauls_store.db"):
        self.hauls_store = HaulsStore(store_path) if HAULS_STORE_AVAILABLE else None
        self.max_context_length = 2000  # Maximum context to add to prompts
    
    def add_training_knowledge(self, dataset_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Add dataset information to knowledge base"""
        if not self.hauls_store:
            return False
        
        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                return False
            
            # Read and process dataset
            if dataset_path.suffix == '.jsonl':
                with open(dataset_path, 'r') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                content = self._extract_content_from_data(data)
                                if content:
                                    doc_metadata = {
                                        **(metadata or {}),
                                        'source': 'training_dataset',
                                        'dataset_name': dataset_path.name,
                                        'line_number': i
                                    }
                                    self.hauls_store.add_document(content, doc_metadata)
                            except json.JSONDecodeError:
                                continue
            else:
                # Handle text files
                with open(dataset_path, 'r') as f:
                    content = f.read()
                    if content:
                        doc_metadata = {
                            **(metadata or {}),
                            'source': 'training_dataset',
                            'dataset_name': dataset_path.name
                        }
                        self.hauls_store.add_document(content, doc_metadata)
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding training knowledge: {e}")
            return False
    
    def _extract_content_from_data(self, data: Dict[str, Any]) -> str:
        """Extract meaningful content from dataset entry"""
        # Try different common content keys
        content_keys = ['text', 'content', 'instruction', 'input', 'prompt', 'conversation']
        
        for key in content_keys:
            if key in data and data[key]:
                content = str(data[key])
                if isinstance(data.get('response'), str) and data['response']:
                    content += f"\nResponse: {data['response']}"
                elif isinstance(data.get('output'), str) and data['output']:
                    content += f"\nOutput: {data['output']}"
                return content
        
        # Fallback: join all string values
        string_values = []
        for value in data.values():
            if isinstance(value, str) and len(value) > 10:
                string_values.append(value)
        
        return "\n".join(string_values) if string_values else ""
    
    def retrieve_context(self, query: str, max_results: int = 3, filter_source: Optional[str] = None) -> str:
        """Retrieve relevant context for query"""
        if not self.hauls_store:
            return ""
        
        # Build filter for specific source if provided
        filter_metadata = {'source': filter_source} if filter_source else None
        
        # Search for relevant documents
        results = self.hauls_store.search(query, top_k=max_results, filter_metadata=filter_metadata)
        
        if not results:
            return ""
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content']
            if current_length + len(content) > self.max_context_length:
                # Truncate to fit within limit
                remaining = self.max_context_length - current_length
                content = content[:remaining] + "..."
            
            context_parts.append(f"[Source: {result['metadata'].get('source', 'unknown')}]\n{content}")
            current_length += len(content)
            
            if current_length >= self.max_context_length:
                break
        
        return "\n\n---\n\n".join(context_parts)
    
    def enhance_training_prompt(self, original_prompt: str, dataset_name: Optional[str] = None) -> str:
        """Enhance training prompt with relevant knowledge"""
        if not self.hauls_store:
            return original_prompt
        
        # Retrieve relevant context
        context = self.retrieve_context(original_prompt, filter_source='training_dataset')
        
        if not context:
            return original_prompt
        
        # Build enhanced prompt
        enhanced_prompt = f"""You are SLO, an AI language model trained on diverse datasets. Use the following relevant knowledge to inform your response:

Relevant Knowledge:
{context}

Original Prompt:
{original_prompt}"""
        
        return enhanced_prompt
    
    def add_conversation_memory(self, conversation_history: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None):
        """Store conversation history in memory"""
        if not self.hauls_store:
            return False
        
        for message in conversation_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            if content:
                full_content = f"Role: {role}\nContent: {content}"
                doc_metadata = {
                    **(metadata or {}),
                    'source': 'conversation',
                    'role': role
                }
                self.hauls_store.add_document(full_content, doc_metadata)
        
        return True
    
    def search_knowledge(self, query: str, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        if not self.hauls_store:
            return []
        
        filter_metadata = {'source': source_filter} if source_filter else None
        return self.hauls_store.search(query, filter_metadata=filter_metadata)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not self.hauls_store:
            return {'error': 'HaulsStore not available'}
        
        stats = self.hauls_store.get_stats()
        
        # Add breakdown by source
        conn = str(self.hauls_store.db_path)
        import sqlite3
        db_conn = sqlite3.connect(conn)
        cursor = db_conn.cursor()
        
        try:
            cursor.execute('''
                SELECT json_extract(metadata, '$.source') as source, COUNT(*) as count
                FROM documents
                WHERE json_extract(metadata, '$.source') IS NOT NULL
                GROUP BY source
            ''')
            source_breakdown = dict(cursor.fetchall())
            stats['source_breakdown'] = source_breakdown
        except:
            stats['source_breakdown'] = {}
        
        db_conn.close()
        return stats
    
    def export_knowledge(self, output_path: str, source_filter: Optional[str] = None):
        """Export knowledge base to file"""
        if not self.hauls_store:
            return False
        
        filter_metadata = {'source': source_filter} if source_filter else None
        docs = self.hauls_store.list_documents(limit=1000)
        
        # Apply filter if needed
        if source_filter:
            filtered_docs = []
            for doc in docs:
                if doc['metadata'].get('source') == source_filter:
                    filtered_docs.append(doc)
            docs = filtered_docs
        
        # Export to file
        try:
            with open(output_path, 'w') as f:
                json.dump(docs, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"❌ Error exporting knowledge: {e}")
            return False


def main():
    """CLI for RAG operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SLO RAG System')
    parser.add_argument('--add-dataset', help='Add dataset to knowledge base')
    parser.add_argument('--search', help='Search knowledge base')
    parser.add_argument('--stats', action='store_true', help='Show knowledge statistics')
    parser.add_argument('--export', help='Export knowledge base to file')
    parser.add_argument('--source-filter', help='Filter by source type')
    parser.add_argument('--store-path', default='runs/store/hauls_store.db', help='HaulsStore path')
    
    args = parser.parse_args()
    
    rag = SLO_RAG(args.store_path)
    
    if args.add_dataset:
        success = rag.add_training_knowledge(args.add_dataset)
        if success:
            print(f"✅ Added dataset to knowledge base: {args.add_dataset}")
        else:
            print(f"❌ Failed to add dataset: {args.add_dataset}")
    
    elif args.search:
        results = rag.search_knowledge(args.search, args.source_filter)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Relevance: {result['similarity']:.2f}]")
            print(f"   {result['content'][:200]}...")
    
    elif args.stats:
        stats = rag.get_knowledge_stats()
        print("🧠 SLO Knowledge Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.export:
        success = rag.export_knowledge(args.export, args.source_filter)
        if success:
            print(f"✅ Exported knowledge to: {args.export}")
        else:
            print(f"❌ Failed to export to: {args.export}")
    
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()