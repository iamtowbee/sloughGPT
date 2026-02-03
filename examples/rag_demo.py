#!/usr/bin/env python3
"""
SloughGPT RAG (Retrieval-Augmented Generation) Demo
Demonstrates knowledge base integration with semantic search
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation System"""
    
    def __init__(self):
        self.knowledge_base = []
        self.embeddings = []
        
    async def initialize(self):
        """Initialize RAG system"""
        try:
            from sloughgpt.data_learning import DatasetPipeline, LearningConfigAdvanced
            from sloughgpt.neural_network import SloughGPT
            
            # Initialize learning pipeline with knowledge retrieval
            config = LearningConfigAdvanced(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                similarity_threshold=0.7,
                chunk_size=512
            )
            
            self.pipeline = DatasetPipeline(config)
            self.model = None  # Would be SloughGPT() in real implementation
            
            logger.info("✅ RAG system initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️  RAG dependencies not available: {e}")
            logger.info("Running in demo mode with mock knowledge base")
            self.pipeline = None
            self.model = None
            self._setup_demo_knowledge()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            return False
    
    def _setup_demo_knowledge(self):
        """Setup demo knowledge base"""
        self.knowledge_base = [
            {
                "id": "kb_001",
                "title": "SloughGPT Architecture",
                "content": "SloughGPT is built with a modular architecture consisting of neural networks, database management, security systems, performance optimization, API servers, and training frameworks. The system supports distributed training, real-time serving, and autonomous learning capabilities.",
                "source": "documentation",
                "created_at": "2024-01-15T10:00:00Z"
            },
            {
                "id": "kb_002", 
                "title": "Training Process",
                "content": "SloughGPT training involves multiple stages including data preprocessing, model initialization, forward/backward passes, gradient clipping, learning rate scheduling, and checkpointing. The system supports both single-GPU and multi-GPU training with automatic scaling.",
                "source": "documentation",
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": "kb_003",
                "title": "Security Features",
                "content": "SloughGPT includes comprehensive security features such as input validation, XSS/SQL injection prevention, rate limiting, authentication, authorization, API key management, and audit logging. All data is encrypted both in transit and at rest.",
                "source": "documentation", 
                "created_at": "2024-01-15T11:00:00Z"
            },
            {
                "id": "kb_004",
                "title": "Performance Optimization",
                "content": "The system employs multiple optimization strategies including model quantization, response caching, request batching, auto-scaling, and memory management. Performance metrics are tracked in real-time with Prometheus/Grafana dashboards.",
                "source": "documentation",
                "created_at": "2024-01-15T11:30:00Z"
            },
            {
                "id": "kb_005",
                "title": "Cost Management",
                "content": "SloughGPT provides comprehensive cost tracking including token-based pricing, storage costs, compute time costs, and network bandwidth usage. Users can set budgets, receive alerts, and get optimization recommendations to control expenses.",
                "source": "documentation",
                "created_at": "2024-01-15T12:00:00Z"
            }
        ]
        
        logger.info(f"📚 Loaded {len(self.knowledge_base)} knowledge items")
    
    async def add_knowledge(self, content: str, title: str = None, source: str = "user"):
        """Add knowledge to the base"""
        knowledge_item = {
            "id": f"kb_{len(self.knowledge_base) + 1:06d}",
            "title": title or f"Knowledge {len(self.knowledge_base) + 1}",
            "content": content,
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.knowledge_base.append(knowledge_item)
        logger.info(f"📝 Added knowledge: {knowledge_item['title']}")
        
        return knowledge_item
    
    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        if not self.knowledge_base:
            return []
        
        # Simple keyword-based search for demo
        # In real implementation, this would use semantic search with embeddings
        query_words = query.lower().split()
        scored_items = []
        
        for item in self.knowledge_base:
            content_lower = item["content"].lower()
            title_lower = item["title"].lower()
            
            # Calculate relevance score
            score = 0
            for word in query_words:
                if word in title_lower:
                    score += 2  # Title matches are more important
                if word in content_lower:
                    score += 1
            
            if score > 0:
                scored_items.append({
                    **item,
                    "relevance_score": score,
                    "matched_words": [w for w in query_words if w in title_lower or w in content_lower]
                })
        
        # Sort by relevance score
        scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_items[:limit]
    
    async def generate_response_with_rag(self, query: str, retrieved_knowledge: List[Dict]) -> str:
        """Generate response using retrieved knowledge"""
        if not self.model:
            # Demo mode with mock responses
            return self._generate_mock_rag_response(query, retrieved_knowledge)
        
        # Real RAG implementation would:
        # 1. Format retrieved knowledge as context
        # 2. Create prompt with context and question
        # 3. Generate response using the model
        # 4. Include citations
        
        context = self._format_context(retrieved_knowledge)
        prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        # This would call the actual model
        response = await self._call_model(prompt)
        return response
    
    def _format_context(self, knowledge_items: List[Dict]) -> str:
        """Format knowledge items as context"""
        context_parts = []
        for i, item in enumerate(knowledge_items, 1):
            context_parts.append(f"[{i}] {item['title']}: {item['content']}")
        
        return "\n\n".join(context_parts)
    
    async def _call_model(self, prompt: str) -> str:
        """Call the actual model"""
        # Placeholder for real model call
        return "This would be the actual model response with context."
    
    def _generate_mock_rag_response(self, query: str, knowledge: List[Dict]) -> str:
        """Generate mock RAG response for demo"""
        if not knowledge:
            return "I don't have specific information about that in my knowledge base. Could you please provide more context or ask about SloughGPT's features?"
        
        # Use the most relevant knowledge item
        best_item = knowledge[0]
        
        responses = {
            "architecture": "SloughGPT has a modular architecture with neural networks, database management, security systems, performance optimization, API servers, and training frameworks.",
            "training": "SloughGPT training involves data preprocessing, model initialization, forward/backward passes, gradient clipping, learning rate scheduling, and checkpointing.",
            "security": "SloughGPT includes input validation, XSS/SQL injection prevention, rate limiting, authentication, authorization, and encryption.",
            "performance": "The system uses model quantization, response caching, request batching, auto-scaling, and real-time performance monitoring.",
            "cost": "SloughGPT tracks token pricing, storage costs, compute time, and bandwidth usage with budgets and optimization recommendations."
        }
        
        # Find best matching response based on keywords
        query_lower = query.lower()
        
        for keyword, response in responses.items():
            if keyword in query_lower:
                citations = f"\n\n📚 Sources: {best_item['title']} ({best_item['source']})"
                return response + citations
        
        # Default response using the knowledge content
        return f"{best_item['content']}\n\n📚 Source: {best_item['title']} ({best_item['source']})"

class RAGCLI:
    """Command-line interface for RAG system"""
    
    def __init__(self):
        self.rag = RAGSystem()
        self.running = True
        
    async def start(self):
        """Start the RAG CLI interface"""
        print("🧠 SloughGPT RAG (Retrieval-Augmented Generation) Demo")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        # Initialize RAG system
        if not await self.rag.initialize():
            print("❌ Failed to initialize RAG system")
            return
        
        print("✅ RAG system ready with knowledge base")
        print()
        
        # Main interaction loop
        while self.running:
            try:
                # Get user input
                user_input = input("🔍 Ask me anything: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.startswith('/'):
                    await self.process_command(user_input)
                else:
                    # Perform RAG query
                    await self.query_rag(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    async def process_command(self, command: str):
        """Process RAG commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/quit" or cmd == "/exit":
            self.running = False
        elif cmd == "/add":
            if len(parts) > 1:
                # Get content after /add
                content = " ".join(parts[1:])
                await self.rag.add_knowledge(content)
                print("✅ Knowledge added successfully!")
            else:
                print("❌ Please provide content to add")
        elif cmd == "/search":
            if len(parts) > 1:
                query = " ".join(parts[1:])
                results = self.rag.search_knowledge(query)
                self.display_search_results(results)
            else:
                print("❌ Please provide a search query")
        elif cmd == "/list":
            self.display_knowledge_base()
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type '/help' for available commands")
    
    async def query_rag(self, query: str):
        """Perform RAG query"""
        print(f"\n🔍 Searching for: {query}")
        
        # Retrieve relevant knowledge
        retrieved = self.rag.search_knowledge(query, limit=3)
        
        if retrieved:
            print(f"📚 Found {len(retrieved)} relevant knowledge items:")
            for i, item in enumerate(retrieved, 1):
                print(f"   {i}. {item['title']} (Score: {item['relevance_score']})")
                print(f"      {item['content'][:100]}...")
            
            print(f"\n🤖 Generating response with RAG...")
            
            # Generate response using retrieved knowledge
            response = await self.rag.generate_response_with_rag(query, retrieved)
            print(f"\n💡 Response:")
            print(response)
        else:
            print("📭 No relevant knowledge found")
            print("🤖 Response: I don't have information about that in my knowledge base.")
        
        print()
    
    def display_search_results(self, results: List[Dict]):
        """Display search results"""
        if not results:
            print("📭 No results found")
            return
        
        print(f"📋 Search Results ({len(results)} items):")
        for i, item in enumerate(results, 1):
            print(f"\n{i}. {item['title']} (Score: {item['relevance_score']})")
            print(f"   📄 Content: {item['content'][:150]}...")
            print(f"   📂 Source: {item['source']}")
            print(f"   🔍 Matched: {', '.join(item['matched_words'])}")
    
    def display_knowledge_base(self):
        """Display all knowledge items"""
        if not self.rag.knowledge_base:
            print("📭 Knowledge base is empty")
            return
        
        print(f"📚 Knowledge Base ({len(self.rag.knowledge_base)} items):")
        for i, item in enumerate(self.rag.knowledge_base, 1):
            print(f"\n{i}. {item['title']}")
            print(f"   📄 {item['content'][:100]}...")
            print(f"   📂 Source: {item['source']}")
            print(f"   📅 Created: {item['created_at']}")
    
    def show_help(self):
        """Show help information"""
        print("📚 RAG Commands:")
        print("   /help              - Show this help message")
        print("   /quit, /exit       - Exit the application")
        print("   /add <content>     - Add knowledge to the base")
        print("   /search <query>     - Search knowledge base")
        print("   /list              - List all knowledge items")
        print()
        print("💡 How RAG Works:")
        print("   1. Your question is searched in the knowledge base")
        print("   2. Relevant information is retrieved")
        print("   3. Response is generated using the retrieved context")
        print("   4. Sources are cited for transparency")

async def main():
    """Main function for RAG demo"""
    parser = argparse.ArgumentParser(description="SloughGPT RAG Demo")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--add", nargs="+", help="Add knowledge to the base")
    parser.add_argument("--search", help="Search knowledge base")
    
    args = parser.parse_args()
    
    # Start CLI
    cli = RAGCLI()
    
    # Handle command-line args
    if args.add:
        content = " ".join(args.add)
        await cli.rag.add_knowledge(content)
        print(f"✅ Added: {content}")
        return
    
    if args.search:
        results = cli.rag.search_knowledge(args.search)
        cli.display_search_results(results)
        return
    
    # Start interactive mode
    await cli.start()

if __name__ == "__main__":
    asyncio.run(main())