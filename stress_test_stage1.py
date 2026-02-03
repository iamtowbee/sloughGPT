#!/usr/bin/env python3
"""
Stage 1 Stress Testing - Real-world scenarios
"""

import sys
import time
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from hauls_store import HaulsStore
from slo_rag import SLO_RAG

def stress_test_performance():
    """Test performance with realistic load"""
    print("🔥 STRESS TESTING Stage 1")
    print("=" * 50)
    
    # Create fresh store
    store = HaulsStore("test_stress.db")
    rag = SLO_RAG("test_stress.db")
    
    print("1. Adding 5000 diverse documents...")
    start_time = time.time()
    
    # Mix of different content types
    document_types = [
        ("shakespeare", "Shakespearean dialogue: To be or not to be, that is the question."),
        ("technical", "Machine learning algorithms require careful hyperparameter tuning for optimal performance."),
        ("conversational", "Hello! How can I assist you today with your inquiry?"),
        ("creative", "The sunset painted the sky in brilliant hues of orange and purple."),
        ("factual", "The Earth has approximately 8 billion inhabitants as of 2023."),
        ("code", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("poetry", "Roses are red, violets are blue, code runs sweet when compiled for you."),
        ("philosophy", "The nature of consciousness remains one of the greatest mysteries."),
        ("news", "Local startup raises $10M in Series A funding for AI technology."),
        ("science", "Quantum computing promises to revolutionize computational capabilities.")
    ]
    
    doc_ids = []
    for i in range(5000):
        doc_type, content = random.choice(document_types)
        doc_id = store.add_document(content, {
            'type': doc_type,
            'index': i,
            'stress_test': True
        })
        doc_ids.append(doc_id)
        
        if i % 500 == 0:
            print(f"  Added {i} documents...")
    
    add_time = time.time() - start_time
    print(f"✅ Added 5000 documents in {add_time:.2f}s ({5000/add_time:.1f} docs/sec)")
    
    print("\n2. Testing search performance...")
    search_queries = [
        "Shakespeare", "machine learning", "conversation", "creative writing",
        "population", "programming", "poetry", "philosophy", "technology",
        "quantum", "assistance", "algorithms", "consciousness", "funding"
    ]
    
    search_times = []
    for query in search_queries:
        for _ in range(10):  # 10 searches per query
            start = time.time()
            results = store.search(query, top_k=5)
            search_time = time.time() - start
            search_times.append(search_time)
    
    avg_search = sum(search_times) / len(search_times)
    max_search = max(search_times)
    min_search = min(search_times)
    
    print(f"✅ Search performance:")
    print(f"  Average: {avg_search*1000:.2f}ms")
    print(f"  Max: {max_search*1000:.2f}ms")
    print(f"  Min: {min_search*1000:.2f}ms")
    
    print("\n3. Testing concurrent operations...")
    start_time = time.time()
    
    # Simulate concurrent add/search operations
    for i in range(100):
        # Add document
        store.add_document(f"Concurrent test document {i}", {'concurrent': True, 'index': i})
        
        # Search for recent document
        store.search(f"concurrent test {i}")
    
    concurrent_time = time.time() - start_time
    print(f"✅ 100 concurrent add/search operations in {concurrent_time:.2f}s")
    
    print("\n4. Testing memory usage...")
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Add many large documents
    for i in range(100):
        large_content = f"Large document {i} " * 1000  # ~15KB each
        store.add_document(large_content, {'large': True, 'index': i})
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before
    
    print(f"✅ Memory usage:")
    print(f"  Before: {memory_before:.1f}MB")
    print(f"  After: {memory_after:.1f}MB") 
    print(f"  Increase: {memory_increase:.1f}MB for 100 large docs")
    
    print("\n5. Testing data consistency...")
    consistency_errors = 0
    
    # Add document, immediately retrieve and verify
    test_content = f"Consistency test {random.randint(10000, 99999)}"
    doc_id = store.add_document(test_content, {'consistency': True})
    
    retrieved = store.get_document(doc_id)
    if retrieved['content'] != test_content:
        consistency_errors += 1
    
    # Search should find the document
    search_results = store.search("consistency test")
    if not any(r['content'] == test_content for r in search_results):
        consistency_errors += 1
    
    print(f"✅ Consistency errors: {consistency_errors}")
    
    print("\n6. Testing RAG integration...")
    rag_start = time.time()
    
    # Create test dataset
    test_dataset = []
    for i in range(500):
        test_dataset.append({
            "instruction": f"Test instruction {i}",
            "content": f"Test content about topic {i} with detailed information",
            "type": "rag_test"
        })
    
    dataset_path = Path("rag_test_dataset.jsonl")
    with open(dataset_path, 'w') as f:
        for item in test_dataset:
            f.write(json.dumps(item) + '\n')
    
    # Add to RAG
    rag.add_training_knowledge(str(dataset_path))
    
    # Test RAG retrieval
    rag_context = rag.retrieve_context("topic 123", max_results=3)
    
    rag_time = time.time() - rag_start
    print(f"✅ RAG integration:")
    print(f"  Added 500 documents to RAG in {rag_time:.2f}s")
    print(f"  Context retrieval: {len(rag_context)} characters")
    
    # Cleanup
    import os
    if dataset_path.exists():
        os.remove(dataset_path)
    
    # Get final stats
    final_stats = store.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Total documents: {final_stats['total_documents']}")
    print(f"  Embedding dimension: {final_stats['embedding_dimension']}")
    print(f"  Endic index size: {final_stats['endic_index_size']}")
    
    # Overall assessment
    print(f"\n🎯 STRESS TEST RESULTS:")
    
    performance_score = 0
    if add_time < 30:  # 5000 docs in < 30s
        performance_score += 20
        print(f"  ✅ Add Performance: {add_time:.1f}s (<30s) [+20]")
    else:
        print(f"  ❌ Add Performance: {add_time:.1f}s (>30s) [+0]")
    
    if avg_search < 0.05:  # < 50ms
        performance_score += 20
        print(f"  ✅ Search Performance: {avg_search*1000:.1f}ms (<50ms) [+20]")
    else:
        print(f"  ❌ Search Performance: {avg_search*1000:.1f}ms (>50ms) [+0]")
    
    if concurrent_time < 10:  # < 10s
        performance_score += 20
        print(f"  ✅ Concurrent Performance: {concurrent_time:.1f}s (<10s) [+20]")
    else:
        print(f"  ❌ Concurrent Performance: {concurrent_time:.1f}s (>10s) [+0]")
    
    if memory_increase < 200:  # < 200MB
        performance_score += 20
        print(f"  ✅ Memory Efficiency: {memory_increase:.1f}MB (<200MB) [+20]")
    else:
        print(f"  ❌ Memory Efficiency: {memory_increase:.1f}MB (>200MB) [+0]")
    
    if consistency_errors == 0:
        performance_score += 20
        print(f"  ✅ Data Consistency: {consistency_errors} errors [+20]")
    else:
        print(f"  ❌ Data Consistency: {consistency_errors} errors [+0]")
    
    total_score = performance_score
    if total_score >= 100:
        print(f"\n🏆 EXCELLENT: {total_score}/100 - Stage 1 is production ready!")
    elif total_score >= 80:
        print(f"\n✅ GOOD: {total_score}/100 - Stage 1 needs minor optimizations")
    elif total_score >= 60:
        print(f"\n⚠️ OK: {total_score}/100 - Stage 1 needs performance improvements")
    else:
        print(f"\n❌ POOR: {total_score}/100 - Stage 1 needs significant work")
    
    return total_score >= 80

if __name__ == "__main__":
    success = stress_test_performance()
    sys.exit(0 if success else 1)