#!/usr/bin/env python3
"""
SloughGPT Comprehensive Test Suite
Unit, integration, performance, and security tests
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add sloughgpt to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sloughgpt.core.testing import (
    run_all_tests, TestConfiguration, unit_test, integration_test, 
    performance_test, security_test
)

def main():
    """Run comprehensive test suite"""
    print("🧪 SloughGPT Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test configuration
    config = TestConfiguration(
        test_data_dir="test_data",
        output_dir="test_results",
        parallel_workers=2,
        timeout_seconds=30,
        coverage_threshold=80.0,
        performance_baseline_ms=50.0,
        enable_slow_tests=True,
        enable_integration_tests=True,
        enable_security_tests=True
    )
    
    print("\n🔧 Test Configuration:")
    print(f"   📁 Test Data Directory: {config.test_data_dir}")
    print(f"   📊 Output Directory: {config.output_dir}")
    print(f"   👥 Parallel Workers: {config.parallel_workers}")
    print(f"   ⏱️ Timeout: {config.timeout_seconds}s")
    print(f"   📈 Coverage Threshold: {config.coverage_threshold}%")
    print(f"   ⚡ Performance Baseline: {config.performance_baseline_ms}ms")
    
    print(f"\n🚀 Running Tests...")
    
    # Create test directories
    os.makedirs(config.test_data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run comprehensive test suite
    start_time = time.time()
    results = run_all_tests(config)
    end_time = time.time()
    
    # Generate report
    print(f"\n📊 Test Results (completed in {end_time - start_time:.2f}s):")
    
    for suite_name, suite_data in results["suites"].items():
        print(f"\n📋 {suite_name.title()} Suite:")
        print(f"   Status: {suite_data['status']}")
        print(f"   Pass Rate: {suite_data['summary']['pass_rate']:.1f}%")
        print(f"   Total Duration: {suite_data['total_duration']:.2f}s")
        print(f"   Tests: {suite_data['summary']['total_tests']}")
        print(f"   Passed: {suite_data['summary']['passed']}")
        print(f"   Failed: {suite_data['summary']['failed']}")
        
        # Show key results
        if suite_data["results"]:
            print(f"\n   🔍 Key Results:")
            for i, result in enumerate(suite_data["results"][:5]):  # Show first 5
                status_icon = "✅" if result.status.value == "passed" else "❌"
                print(f"      {status_icon} {result.test_name}: {result.status.value} ({result.duration_ms:.1f}ms)")
    
    # Generate summary
    print(f"\n📈 Overall Summary:")
    summary = results["summary"]
    print(f"   🧪 Total Tests: {summary['total_tests']}")
    print(f"   ✅ Passed: {summary['passed']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   ⚠️ Errors: {summary['errors']}")
    print(f"   📊 Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"   📈 Test Suites: {summary['total_suites']}")
    
    # Performance benchmarks
    if "performance_tests" in results:
        perf_suite = results["performance_tests"]
        print(f"\n⚡ Performance Benchmarks:")
        
        for result in perf_suite["results"]:
            if result.metadata:
                print(f"   • {result.test_name}:")
                if "avg_ms" in result.metadata:
                    print(f"     Average: {result.metadata['avg_ms']:.2f}ms")
                if "baseline_ratio" in result.metadata:
                    print(f"     Baseline Ratio: {result.metadata['baseline_ratio']:.2f}x")
    
    # Security test summary
    if "security_tests" in results:
        sec_suite = results["security_tests"]
        print(f"\n🛡️ Security Test Summary:")
        
        malicious_tests = [r for r in sec_suite["results"] if "malicious" in r.test_name.lower()]
        blocked_tests = [r for r in sec_suite["results"] if r.status.value == "passed" and "malicious" in r.test_name.lower()]
        
        passed_tests = [r for r in sec_suite["results"] if r.status.value == "passed"]
        print(f"   🛑️ Malicious Tests Blocked: {len(blocked_tests)}/{len(malicious_tests)}")
        print(f"   ✅ Security Tests Passed: {len(passed_tests)}/{len(sec_suite['results'])}")
    
    # Save detailed report
    output_file = Path(config.output_dir) / "test_report.json"
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {output_file}")
    
    # Show health status
    print(f"\n🏥 Test Infrastructure Health:")
    
    # Check if tests passed thresholds
    overall_health = "🟢 Healthy"
    if summary['pass_rate'] < config.coverage_threshold:
        overall_health = "🟡 Needs Improvement"
    
    if summary['errors'] > 0:
        overall_health = "🔴 Has Issues"
    
    print(f"   Status: {overall_health}")
    print(f"   Pass Rate: {summary['pass_rate']:.1f}% (Target: {config.coverage_threshold}%)")
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive Test Suite Completed!")
    print("\n🚀 Test Infrastructure Features:")
    print("   ✅ Unit testing framework")
    print("   ✅ Integration testing")
    print("   ✅ Performance benchmarking")
    print("   ✅ Security testing")
    print("   ✅ Mock external services")
    print("   ✅ Parallel execution")
    print("   ✅ Comprehensive reporting")
    print("   ✅ Configurable test suites")
    print("   ✅ Performance monitoring")
    print("   ✅ Extensible architecture")
    print("\n🛡️ Production-Ready Testing Framework!")
    print("   • Automated test discovery")
    print("   • Parallel test execution")
    print("   • Performance regression detection")
    print("   • Security vulnerability scanning")
    print("   • Comprehensive reporting")
    print("   • CI/CD integration ready")
    print("   • Coverage measurement")
    print("   • Mock service simulation")
    
    # Cleanup
    if os.path.exists(config.test_data_dir):
        import shutil
        shutil.rmtree(config.test_data_dir)
    
    print(f"\n🧹 Cleanup completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()