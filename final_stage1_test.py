#!/usr/bin/env python3
"""
Final Stage 1 Performance Test
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from hauls_store import HaulsStore

def final_performance_test():
    print("🏁 Final Stage 1 Performance Test")
    print("=" * 50)
    
    # Test with optimized HaulsStore
    store = HaulsStore("final_performance_test.db")
    
    # Test 1: Batch addition performance
    print("1. Testing 2000 document batch addition...")
    start_time = time.time()
    
    doc_ids = []
    for i in range(2000):
        content = f"Performance test document {i} with unique content for final validation"
        doc_id = store.add_document(content, {'final_test': True, 'index': i})
        doc_ids.append(doc_id)
    
    add_time = time.time() - start_time
    docs_per_sec = len(doc_ids) / add_time
    
    print(f"✅ Added {len(doc_ids)} documents in {add_time:.2f}s ({docs_per_sec:.1f} docs/sec)")
    
    # Performance threshold check
    if docs_per_sec >= 200:
        print("  🎯 EXCELLENT: Meets performance target (>200 docs/sec)")
    elif docs_per_sec >= 100:
        print("  ✅ GOOD: Above minimum threshold (>100 docs/sec)")
    else:
        print("  ⚠️ NEEDS WORK: Below performance threshold")
    
    # Test 2: Search performance under load
    print("\n2. Testing search performance with 2000+ documents...")
    search_times = []
    
    for i in range(50):
        start_time = time.time()
        results = store.search(f"performance test {i}", top_k=5)
        search_time = time.time() - start_time
        search_times.append(search_time)
    
    avg_search = sum(search_times) / len(search_times)
    max_search = max(search_times)
    min_search = min(search_times)
    
    print(f"✅ Search Performance (50 queries):")
    print(f"  Average: {avg_search*1000:.2f}ms")
    print(f"  Max: {max_search*1000:.2f}ms")
    print(f"  Min: {min_search*1000:.2f}ms")
    
    # Search performance threshold check
    if avg_search <= 0.05:  # 50ms
        print("  🎯 EXCELLENT: Meets search target (<50ms)")
    elif avg_search <= 0.1:  # 100ms
        print("  ✅ GOOD: Acceptable search speed (<100ms)")
    else:
        print("  ⚠️ NEEDS WORK: Search too slow")
    
    # Test 3: Memory efficiency
    print("\n3. Testing memory efficiency...")
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        memory_per_doc = memory_mb / len(store.documents)
        print(f"✅ Memory Usage:")
        print(f"  Total: {memory_mb:.1f}MB")
        print(f"  Per document: {memory_per_doc:.3f}MB")
        print(f"  Document count: {len(store.documents)}")
        
        if memory_per_doc <= 0.01:  # 10KB per doc
            print("  🎯 EXCELLENT: Very memory efficient")
        elif memory_per_doc <= 0.05:  # 50KB per doc
            print("  ✅ GOOD: Reasonable memory usage")
        else:
            print("  ⚠️ NEEDS WORK: High memory usage")
    
    except ImportError:
        print("  ℹ️  psutil not available for memory testing")
    
    # Test 4: Consistency under stress
    print("\n4. Testing consistency under stress...")
    consistency_errors = 0
    
    for i in range(100):
        # Add document
        test_content = f"Stress test consistency {i}"
        doc_id = store.add_document(test_content, {'stress_test': True, 'iteration': i})
        
        # Immediately retrieve and verify
        retrieved = store.get_document(doc_id)
        if not retrieved or retrieved['content'] != test_content:
            consistency_errors += 1
        
        # Search should find the document
        search_results = store.search(f"stress test consistency {i}")
        if not any(r['content'] == test_content for r in search_results):
            consistency_errors += 1
    
    consistency_rate = (100 - consistency_errors) / 100 * 100
    print(f"✅ Consistency Rate: {consistency_rate:.1f}% ({consistency_errors} errors)")
    
    if consistency_rate >= 99.5:
        print("  🎯 EXCELLENT: Near-perfect consistency")
    elif consistency_rate >= 95:
        print("  ✅ GOOD: Reliable consistency")
    else:
        print("  ⚠️ NEEDS WORK: Consistency issues")
    
    # Final stats
    stats = store.get_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Overall assessment
    print(f"\n🏆 OVERALL STAGE 1 ASSESSMENT:")
    performance_score = 0
    
    # Scoring
    if docs_per_sec >= 200:
        performance_score += 25
    elif docs_per_sec >= 100:
        performance_score += 15
    else:
        performance_score += 5
    
    if avg_search <= 0.05:
        performance_score += 25
    elif avg_search <= 0.1:
        performance_score += 15
    else:
        performance_score += 5
    
    try:
        if memory_per_doc <= 0.01:
            performance_score += 25
        elif memory_per_doc <= 0.05:
            performance_score += 15
        else:
            performance_score += 5
    except:
        performance_score += 15  # Average score if can't test
    
    if consistency_rate >= 99.5:
        performance_score += 25
    elif consistency_rate >= 95:
        performance_score += 15
    else:
        performance_score += 5
    
    # Final determination
    if performance_score >= 90:
        print(f"🎯 EXCELLENT: {performance_score}/100 - Stage 1 PRODUCTION READY!")
        print("  → Ready for Stage 2: Cognitive Architecture")
    elif performance_score >= 70:
        print(f"✅ GOOD: {performance_score}/100 - Stage 1 ready with minor optimizations")
        print("  → Can proceed to Stage 2 with planned improvements")
    elif performance_score >= 50:
        print(f"⚠️ OK: {performance_score}/100 - Stage 1 needs significant work")
        print("  → Should optimize before Stage 2")
    else:
        print(f"❌ POOR: {performance_score}/100 - Stage 1 requires major redesign")
        print("  → Cannot proceed to Stage 2")
    
    return performance_score >= 70

if __name__ == "__main__":
    success = final_performance_test()
    sys.exit(0 if success else 1)