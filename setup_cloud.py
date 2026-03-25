#!/usr/bin/env python3
"""
Cloud Vector Store Setup Script

This script helps you set up and connect to cloud vector stores.
Run with --help to see all options.

Usage:
    # Test all providers
    python setup_cloud.py --test-all

    # Setup Pinecone
    python setup_cloud.py --provider pinecone --api-key YOUR_KEY --index my-index

    # Setup Weaviate
    python setup_cloud.py --provider weaviate --url https://xxx.weaviate.cloud

    # Test with ChromaDB (no setup required)
    python setup_cloud.py --provider chromadb
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def setup_provider(provider: str, **kwargs):
    """Set up a vector store provider."""
    from domains.inference.vector_store import (
        VectorStoreFactory,
        VectorEntry,
        simple_embed,
    )
    
    print(f"\n{'='*60}")
    print(f"Setting up {provider.upper()} Vector Store")
    print('='*60)
    
    try:
        # Create store
        store = VectorStoreFactory.create(provider)
        print(f"✓ Created {type(store).__name__}")
        
        # Connect
        connected = await store.connect()
        if not connected:
            print(f"✗ Failed to connect")
            return False
        
        print(f"✓ Connected successfully")
        
        # Get dimension
        dim = kwargs.get('dimension', 384)
        
        # Test upsert
        test_entries = [
            VectorEntry(
                id="test_1",
                vector=simple_embed("This is a test document", dimension=dim),
                text="This is a test document for vector store verification",
                metadata={"type": "test", "provider": provider},
            ),
            VectorEntry(
                id="test_2",
                vector=simple_embed("Machine learning is a subset of AI", dimension=dim),
                text="Machine learning is a subset of artificial intelligence",
                metadata={"type": "test", "provider": provider},
            ),
        ]
        
        count = await store.upsert(test_entries)
        print(f"✓ Upserted {count} test documents")
        
        # Test query
        results = await store.query(
            vector=simple_embed("What is machine learning?", dimension=dim),
            top_k=2,
        )
        print(f"✓ Query returned {len(results)} results")
        
        if results:
            print(f"  Top result: {results[0].text[:50]}... (score: {results[0].score:.3f})")
        
        # Stats
        total = await store.count()
        print(f"✓ Total documents: {total}")
        
        # Cleanup test data
        await store.delete(["test_1", "test_2"])
        print(f"✓ Cleaned up test data")
        
        await store.disconnect()
        print(f"✓ Disconnected")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print(f"  Install with: pip install {str(e).split()[-1]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_all_providers():
    """Test all available providers."""
    from domains.inference.vector_store import VectorStoreType
    
    providers = [
        ("chromadb", {"dimension": 384}),
    ]
    
    # Check for cloud providers
    if os.getenv("PINECONE_API_KEY"):
        providers.append(("pinecone", {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "index": os.getenv("PINECONE_INDEX", "sloughgpt"),
            "dimension": 768,
        }))
    
    if os.getenv("WEAVIATE_URL"):
        providers.append(("weaviate", {
            "url": os.getenv("WEAVIATE_URL"),
            "api_key": os.getenv("WEAVIATE_API_KEY"),
            "dimension": 768,
        }))
    
    results = {}
    for provider, kwargs in providers:
        results[provider] = await setup_provider(provider, **kwargs)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for provider, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {provider.upper()}: {status}")
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Set up cloud vector stores for SloughGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ChromaDB (local)
  python setup_cloud.py --provider chromadb

  # Setup Pinecone (requires API key)
  python setup_cloud.py --provider pinecone --api-key sk-xxx

  # Setup Weaviate Cloud
  python setup_cloud.py --provider weaviate --url https://xxx.weaviate.cloud

  # Test all configured providers
  python setup_cloud.py --test-all

Environment variables:
  PINECONE_API_KEY    - Pinecone API key
  PINECONE_INDEX     - Pinecone index name (default: sloughgpt)
  WEAVIATE_URL        - Weaviate URL
  WEAVIATE_API_KEY    - Weaviate API key (optional)
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["chromadb", "pinecone", "weaviate", "in_memory"],
        help="Vector store provider to set up",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the provider (or set PINECONE_API_KEY env var)",
    )
    parser.add_argument(
        "--url",
        help="URL for Weaviate (or set WEAVIATE_URL env var)",
    )
    parser.add_argument(
        "--index",
        default="sloughgpt",
        help="Index name for Pinecone (default: sloughgpt)",
    )
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=384,
        help="Embedding dimension (default: 384)",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all configured providers",
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        success = asyncio.run(test_all_providers())
        sys.exit(0 if success else 1)
    
    if not args.provider:
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START")
        print("="*60)
        print("""
1. ChromaDB (local - no setup):
   python setup_cloud.py --provider chromadb

2. Pinecone (cloud):
   - Get API key from https://app.pinecone.io
   - Set: export PINECONE_API_KEY=your-key
   - Run: python setup_cloud.py --provider pinecone

3. Weaviate (cloud):
   - Get URL from Weaviate Cloud Console
   - Set: export WEAVIATE_URL=https://xxx.weaviate.cloud
   - Run: python setup_cloud.py --provider weaviate
        """)
        sys.exit(0)
    
    # Build kwargs
    kwargs = {"dimension": args.dimension}
    
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.url:
        kwargs["url"] = args.url
    if args.index and args.provider == "pinecone":
        kwargs["index"] = args.index
    
    success = asyncio.run(setup_provider(args.provider, **kwargs))
    
    if success:
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        if args.provider == "chromadb":
            print("  Use in your code:")
            print("    from domains.inference.vector_store import create_vector_store")
            print("    store = await create_vector_store('chromadb')")
        elif args.provider == "pinecone":
            print("  Set in environment:")
            print(f"    export VECTOR_STORE_PROVIDER=pinecone")
            print(f"    export PINECONE_API_KEY={kwargs.get('api_key', 'YOUR_KEY')}")
        else:
            print("  Use the API endpoint to add documents:")
            print("    POST /vector/upsert")
    else:
        print("\n✗ Setup failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
