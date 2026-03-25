"""
Cloud Vector Store Connector

Provides easy setup for production vector stores:
- Pinecone (cloud)
- Weaviate (cloud or self-hosted)
- ChromaDB (local with persistence)

Usage:
    python -m domains.inference.cloud_vector_store --setup pinecone --api-key YOUR_KEY
    python -m domains.inference.cloud_vector_store --setup weaviate --url https://xxx.weaviate.cloud
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def setup_pinecone(api_key: str, index: str = "sloughgpt", dimension: int = 768):
    """Setup Pinecone vector store."""
    from domains.inference.vector_store import PineconeStore, VectorEntry
    
    print(f"Connecting to Pinecone index: {index}")
    store = PineconeStore(
        api_key=api_key,
        index=index,
        dimension=dimension,
    )
    
    connected = await store.connect()
    if connected:
        print("✓ Connected to Pinecone")
        
        # Test with sample data
        entries = [
            VectorEntry(
                id="sample_1",
                vector=[0.1] * dimension,
                text="This is a sample document for testing",
                metadata={"type": "test", "created_by": "cloud_vector_store"},
            ),
        ]
        
        count = await store.upsert(entries)
        print(f"✓ Upserted {count} test documents")
        
        results = await store.query(vector=[0.1] * dimension, top_k=1)
        print(f"✓ Query returned {len(results)} results")
        
        await store.disconnect()
        return True
    else:
        print("✗ Failed to connect to Pinecone")
        return False


async def setup_weaviate(url: str, api_key: str = None, class_name: str = "Document"):
    """Setup Weaviate vector store."""
    from domains.inference.vector_store import WeaviateStore, VectorEntry
    
    print(f"Connecting to Weaviate: {url}")
    store = WeaviateStore(
        url=url,
        api_key=api_key,
        class_name=class_name,
    )
    
    connected = await store.connect()
    if connected:
        print("✓ Connected to Weaviate")
        
        # Test with sample data
        dimension = 768
        entries = [
            VectorEntry(
                id="sample_1",
                vector=[0.1] * dimension,
                text="This is a sample document for testing",
                metadata={"type": "test", "created_by": "cloud_vector_store"},
            ),
        ]
        
        count = await store.upsert(entries)
        print(f"✓ Upserted {count} test documents")
        
        results = await store.query(vector=[0.1] * dimension, top_k=1)
        print(f"✓ Query returned {len(results)} results")
        
        await store.disconnect()
        return True
    else:
        print("✗ Failed to connect to Weaviate")
        return False


async def setup_chromadb(persist_dir: str = "./vector_store", collection: str = "documents"):
    """Setup ChromaDB vector store."""
    from domains.inference.vector_store import ChromaStore, VectorEntry
    
    print(f"Setting up ChromaDB at: {persist_dir}")
    store = ChromaStore(
        persist_directory=persist_dir,
        collection_name=collection,
    )
    
    connected = await store.connect()
    if connected:
        print("✓ Connected to ChromaDB")
        
        # Test with sample data
        dimension = 768
        entries = [
            VectorEntry(
                id="sample_1",
                vector=[0.1] * dimension,
                text="This is a sample document for testing",
                metadata={"type": "test", "created_by": "cloud_vector_store"},
            ),
        ]
        
        count = await store.upsert(entries)
        print(f"✓ Upserted {count} test documents")
        
        results = await store.query(vector=[0.1] * dimension, top_k=1)
        print(f"✓ Query returned {len(results)} results")
        
        await store.disconnect()
        return True
    else:
        print("✗ Failed to connect to ChromaDB")
        return False


async def test_all_providers():
    """Test all vector store providers."""
    from domains.inference.vector_store import (
        InMemoryVectorStore,
        PineconeStore,
        WeaviateStore,
        ChromaStore,
        VectorEntry,
    )
    
    print("\n" + "="*60)
    print("TESTING ALL VECTOR STORE PROVIDERS")
    print("="*60 + "\n")
    
    dimension = 768
    
    # Test In-Memory (always works)
    print("1. Testing In-Memory Store...")
    store = InMemoryVectorStore()
    if await store.connect():
        entries = [VectorEntry(id="1", vector=[0.1]*dimension, text="Test")]
        await store.upsert(entries)
        results = await store.query(vector=[0.1]*dimension, top_k=1)
        print(f"   ✓ In-Memory: {len(results)} results\n")
    
    # Test Pinecone (if API key provided)
    print("2. Testing Pinecone...")
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        if await setup_pinecone(api_key):
            print("   ✓ Pinecone: Connected and tested\n")
    else:
        print("   ⊘ Pinecone: PINECONE_API_KEY not set\n")
    
    # Test Weaviate (if URL provided)
    print("3. Testing Weaviate...")
    url = os.getenv("WEAVIATE_URL")
    if url:
        api_key = os.getenv("WEAVIATE_API_KEY")
        if await setup_weaviate(url, api_key):
            print("   ✓ Weaviate: Connected and tested\n")
    else:
        print("   ⊘ Weaviate: WEAVIATE_URL not set\n")
    
    # Test ChromaDB
    print("4. Testing ChromaDB...")
    if await setup_chromadb():
        print("   ✓ ChromaDB: Connected and tested\n")
    
    print("="*60)
    print("CLOUD SETUP COMPLETE")
    print("="*60)
    print("""
To connect to cloud providers, set these environment variables:

Pinecone:
  export PINECONE_API_KEY="your-api-key"
  python -m domains.inference.cloud_vector_store --setup pinecone

Weaviate:
  export WEAVIATE_URL="https://xxx.weaviate.cloud"
  export WEAVIATE_API_KEY="your-api-key"  # optional
  python -m domains.inference.cloud_vector_store --setup weaviate

Or use the API:
  POST /vector/init
  {
    "provider": "pinecone",
    "api_key": "your-key",
    "index": "production"
  }
""")


def main():
    parser = argparse.ArgumentParser(description="Cloud Vector Store Setup")
    parser.add_argument("--setup", choices=["pinecone", "weaviate", "chromadb", "test-all"],
                        help="Setup provider")
    parser.add_argument("--api-key", help="API key for provider")
    parser.add_argument("--url", help="URL for Weaviate")
    parser.add_argument("--index", default="sloughgpt", help="Pinecone index name")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    
    args = parser.parse_args()
    
    if args.setup == "pinecone":
        api_key = args.api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("Error: Pinecone API key required (--api-key or PINECONE_API_KEY)")
            sys.exit(1)
        asyncio.run(setup_pinecone(api_key, args.index, args.dimension))
    
    elif args.setup == "weaviate":
        url = args.url or os.getenv("WEAVIATE_URL")
        if not url:
            print("Error: Weaviate URL required (--url or WEAVIATE_URL)")
            sys.exit(1)
        api_key = args.api_key or os.getenv("WEAVIATE_API_KEY")
        asyncio.run(setup_weaviate(url, api_key))
    
    elif args.setup == "chromadb":
        asyncio.run(setup_chromadb())
    
    elif args.setup == "test-all":
        asyncio.run(test_all_providers())
    
    else:
        parser.print_help()
        print("\n\nQuick test: python -m domains.inference.cloud_vector_store --setup chromadb")


if __name__ == "__main__":
    main()
