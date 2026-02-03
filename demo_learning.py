#!/usr/bin/env python3
"""
SloughGPT Learning Pipeline Demo
Demonstrates the autonomous data learning capabilities
"""

import asyncio
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_learning_pipeline():
    """Demonstrate the learning pipeline with sample data"""
    
    # Import after ensuring it's available
    try:
        from sloughgpt.data_learning import DatasetPipeline
    except ImportError as e:
        logger.error(f"Learning pipeline not available: {e}")
        logger.info("Install learning dependencies with: pip install -r learning-requirements.txt")
        return
        
    # Create sample data directory
    data_dir = Path("./sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample documentation
    docs_dir = data_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    sample_markdown = """# SloughGPT Documentation

## Introduction
SloughGPT is an advanced transformer-based language model with cognitive capabilities. It can learn from various data sources and improve its performance over time.

## Features
- Multi-step reasoning with chain-of-thought
- Distributed training framework
- Real-time model serving
- Knowledge graph integration
- Autonomous learning capabilities

## Usage
You can use SloughGPT for various natural language processing tasks including text generation, question answering, and content creation.

The model continuously learns from new data and user interactions to improve its performance.
"""
    
    with open(docs_dir / "overview.md", "w") as f:
        f.write(sample_markdown)
    
    # Create sample training data
    training_dir = data_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    sample_training_data = [
        {
            "question": "What is SloughGPT?",
            "answer": "SloughGPT is an advanced transformer-based language model with cognitive capabilities and autonomous learning.",
            "category": "general",
            "difficulty": "easy"
        },
        {
            "question": "How does SloughGPT learn?",
            "answer": "SloughGPT learns through the data learning pipeline that processes various data sources, creates knowledge embeddings, and continuously updates its knowledge base.",
            "category": "learning",
            "difficulty": "medium"
        },
        {
            "question": "What are the main components of SloughGPT?",
            "answer": "SloughGPT consists of neural networks, database management, security systems, performance optimization, API servers, and training frameworks.",
            "category": "architecture",
            "difficulty": "hard"
        }
    ]
    
    with open(training_dir / "qa_data.json", "w") as f:
        json.dump(sample_training_data, f, indent=2)
    
    # Create learning pipeline
    logger.info("🚀 Creating SloughGPT Learning Pipeline...")
    pipeline = DatasetPipeline()
    
    # Add data sources
    logger.info("📚 Adding data sources...")
    pipeline.add_source(
        name="documentation",
        source=str(docs_dir),
        source_type="file",
        format="markdown",
        quality_threshold=0.7
    )
    
    pipeline.add_source(
        name="training_data",
        source=str(training_dir / "qa_data.json"),
        source_type="file", 
        format="json",
        quality_threshold=0.8
    )
    
    # Start learning
    logger.info("🧠 Starting autonomous learning...")
    
    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 256,
        "quality_threshold": 0.75,
        "batch_size": 16
    }
    
    try:
        job_id = await pipeline.start_learning(config)
        logger.info(f"✅ Learning completed! Job ID: {job_id}")
        
        # Show learning statistics
        stats = pipeline.get_learning_stats()
        logger.info("📊 Learning Statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
            
        # Test knowledge search
        logger.info("\n🔍 Testing knowledge search...")
        
        test_queries = [
            "What is SloughGPT?",
            "How does learning work?",
            "Main components of the system"
        ]
        
        for query in test_queries:
            results = pipeline.search_knowledge(query, k=3)
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Results: {len(results)} found")
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result['score']:.3f}")
                logger.info(f"     Text: {result['text'][:100]}...")
                logger.info(f"     Source: {result['metadata'].get('source', 'unknown')}")
                
    except Exception as e:
        logger.error(f"❌ Learning failed: {str(e)}")
        logger.info("This is normal if sentence-transformers or faiss are not installed.")
        logger.info("Install with: pip install -r learning-requirements.txt")
    
    logger.info("\n🎉 Learning Pipeline Demo Complete!")

async def demo_advanced_features():
    """Demonstrate advanced learning features"""
    
    logger.info("🚀 Demonstrating Advanced Learning Features...")
    
    # Create advanced configuration
    from sloughgpt.data_learning import DatasetPipeline, LearningConfigAdvanced
    
    advanced_config = LearningConfigAdvanced(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        max_chunks_per_document=500,
        similarity_threshold=0.8,
        min_chunk_length=30,
        max_chunk_length=800,
        quality_threshold=0.9,
        deduplication_threshold=0.98
    )
    
    pipeline = DatasetPipeline(advanced_config)
    
    # Add web source (simulated)
    logger.info("🌐 Adding web data source...")
    pipeline.add_source(
        name="sample_api",
        source="https://jsonplaceholder.typicode.com/posts/1",
        source_type="api",
        format="json",
        quality_threshold=0.6
    )
    
    logger.info("✅ Advanced features configured successfully!")

if __name__ == "__main__":
    print("🧠 SloughGPT Learning Pipeline Demo")
    print("=" * 50)
    
    # Run demo
    asyncio.run(demo_learning_pipeline())
    
    print("\n" + "=" * 50)
    print("🎯 Try running with real data:")
    print("   python demo_learning.py")
    print("\n📦 Install learning dependencies:")
    print("   pip install -r learning-requirements.txt")