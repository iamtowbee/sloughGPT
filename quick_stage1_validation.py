#!/usr/bin/env python3
"""
Quick Stage 1 Performance Validation
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from hauls_store import HaulsStore

def quick_validation():
    print("🚀 Quick Stage 1 Validation")
    print("=" * 40)
    
    # Fresh store
    store = HaulsStore("quick_validation.db")
    
    # Test 1: Add 100 documents quickly
    print("1. Adding 100 documents...")
    start = time.time()
    
    doc_ids = []
    for i in range(100):
        content = f"Quick test document {i} with content"
        doc_id = store.add_document(content, {'quick_test': True})
        doc_ids.append(doc_id)
    
    add_time = time.time() - start
    docs_per_sec = 100 / add_time
    print(f"   ✅ {100} docs in {add_time:.2f}s ({docs_per_sec:.0f} docs/sec)")
    
    # Test 2: Search performance
    print("2. Testing search...")
    search_times = []
    
    for i in range(10):
        start = time.time()
        results = store.search(f"quick test {i}")
        search_time = time.time() - start
        search_times.append(search_time)
    
    avg_search = sum(search_times) / len(search_times)
    print(f"   ✅ Avg search: {avg_search*1000:.1f}ms")
    
    # Test 3: Memory check
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"   ✅ Memory: {memory_mb:.1f}MB for {len(store.documents)} docs")
    except:
        print("   ℹ️  Memory check not available")
    
    # Test 4: Consistency
    consistency_errors = 0
    for i in range(10):
        test_content = f"Consistency test {i}"
        doc_id = store.add_document(test_content, {'consistency': True})
        retrieved = store.get_document(doc_id)
        if not retrieved or retrieved['content'] != test_content:
            consistency_errors += 1
    
    consistency_rate = (10 - consistency_errors) / 10 * 100
    print(f"   ✅ Consistency: {consistency_rate:.0f}%")
    
    # Final stats
    stats = store.get_stats()
    print(f"\n📊 Final: {stats['total_documents']} docs, "
          f"endic size: {stats['endic_index_size']}")
    
    # Assessment
    score = 0
    if docs_per_sec >= 50:  # Minimum acceptable
        score += 25
        print("   ✅ Add Performance: PASS")
    else:
        print("   ❌ Add Performance: FAIL")
    
    if avg_search <= 0.05:  # Under 50ms
        score += 25
        print("   ✅ Search Performance: PASS")
    else:
        print("   ❌ Search Performance: FAIL")
    
    if consistency_rate >= 90:
        score += 25
        print("   ✅ Consistency: PASS")
    else:
        print("   ❌ Consistency: FAIL")
    
    if stats['endic_index_size'] == stats['total_documents']:
        score += 25
        print("   ✅ Index Sync: PASS")
    else:
        print("   ❌ Index Sync: FAIL")
    
    print(f"\n🏆 Score: {score}/100")
    
    if score >= 75:
        print("🎯 Stage 1: VALIDATED - Ready for Stage 2")
        return True
    else:
        print("⚠️ Stage 1: NEEDS WORK - Fix issues before Stage 2")
        return False

if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)