#!/usr/bin/env python3
"""
Advanced Reasoning Engine Benchmark Suite

Comprehensive testing and performance evaluation system
"""

import time
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from advanced_reasoning_engine import AdvancedReasoningEngine
from hauls_store import Document

@dataclass
class BenchmarkTest:
    """Individual benchmark test case"""
    name: str
    query: str
    expected_pattern: str
    expected_confidence_range: Tuple[float, float]
    description: str

@dataclass 
class BenchmarkResult:
    """Result from benchmark test"""
    test_name: str
    passed: bool
    actual_confidence: float
    reasoning_time: float
    response_quality: float
    error_message: str = ""

class ReasoningBenchmark:
    """Comprehensive benchmark suite for reasoning engine"""
    
    def __init__(self, engine_path: str = "runs/store/hauls_store.db"):
        self.engine = AdvancedReasoningEngine(engine_path)
        self.tests = self._create_test_suite()
        self.results = []
        
    def _create_test_suite(self) -> List[BenchmarkTest]:
        """Create comprehensive test suite"""
        return [
            # Factual queries
            BenchmarkTest(
                name="Factual_Who",
                query="Who is Hamlet in Shakespeare's play?",
                expected_pattern="hybrid",
                expected_confidence_range=(0.7, 0.95),
                description="Basic factual recall test"
            ),
            BenchmarkTest(
                name="Factual_What", 
                query="What are the main themes in Macbeth?",
                expected_pattern="hybrid",
                expected_confidence_range=(0.6, 0.90),
                description="Theme identification test"
            ),
            
            # Analytical queries
            BenchmarkTest(
                name="Analytical_Why",
                query="Why is the balcony scene famous in Romeo and Juliet?",
                expected_pattern="hybrid", 
                expected_confidence_range=(0.7, 0.95),
                description="Causal analysis test"
            ),
            BenchmarkTest(
                name="Analytical_How",
                query="How does Shakespeare develop character motivations in tragedies?",
                expected_pattern="chain_of_thought",
                expected_confidence_range=(0.6, 0.85),
                description="Process analysis test"
            ),
            
            # Complex queries
            BenchmarkTest(
                name="Complex_Compare",
                query="Compare narrative techniques in Hamlet and King Lear",
                expected_pattern="chain_of_thought",
                expected_confidence_range=(0.5, 0.80),
                description="Comparative analysis test"
            ),
            BenchmarkTest(
                name="Complex_Evaluate",
                query="Evaluate the effectiveness of Shakespeare's use of dramatic irony",
                expected_pattern="self_reflective",
                expected_confidence_range=(0.5, 0.80),
                description="Evaluation and reflection test"
            ),
            
            # Performance stress tests
            BenchmarkTest(
                name="Stress_Multi_Hop",
                query="Trace the development of Hamlet's philosophical journey through multiple scenes",
                expected_pattern="multi_hop",
                expected_confidence_range=(0.4, 0.75),
                description="Multi-hop reasoning stress test"
            ),
            
            # Edge cases
            BenchmarkTest(
                name="Edge_Vague",
                query="Tell me about things",
                expected_pattern="hybrid",
                expected_confidence_range=(0.3, 0.60),
                description="Vague query handling test"
            ),
            BenchmarkTest(
                name="Edge_Off_Knowledge",
                query="Analyze quantum physics applications in Elizabethan drama",
                expected_pattern="hybrid",
                expected_confidence_range=(0.2, 0.50),
                description="Out-of-domain query test"
            )
        ]
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🧪 Starting Advanced Reasoning Engine Benchmark Suite")
        print(f"📚 Knowledge Base: {len(self.engine.slo_rag.hauls_store.documents)} documents")
        print(f"🧠 Engine: Enhanced with {len(self.engine.reasoning_patterns)} reasoning patterns")
        print(f"📊 Running {len(self.tests)} comprehensive tests...\n")
        
        # Start performance monitoring
        if hasattr(self.engine, 'start_performance_monitoring'):
            self.engine.start_performance_monitoring()
        
        total_start = time.time()
        
        # Run each test
        for i, test in enumerate(self.tests, 1):
            print(f"🔍 Test {i}/{len(self.tests)}: {test.name}")
            print(f"   Query: {test.query}")
            print(f"   Description: {test.description}")
            
            result = self._run_single_test(test)
            self.results.append(result)
            
            if result.passed:
                print(f"   ✅ PASSED - Confidence: {result.actual_confidence:.3f}, Time: {result.reasoning_time:.3f}s")
            else:
                print(f"   ❌ FAILED - {result.error_message}")
            print()
        
        total_time = time.time() - total_start
        
        # Calculate benchmark summary
        summary = self._calculate_summary(total_time)
        
        # Print results
        self._print_results(summary)
        
        return summary
    
    def _run_single_test(self, test: BenchmarkTest) -> BenchmarkResult:
        """Run single benchmark test"""
        try:
            start_time = time.time()
            
            # Run reasoning
            reasoning_chain = self.engine.reason(test.query, test.expected_pattern)
            
            reasoning_time = time.time() - start_time
            confidence = reasoning_chain.total_confidence
            
            # Evaluate test criteria
            confidence_in_range = test.expected_confidence_range[0] <= confidence <= test.expected_confidence_range[1]
            response_quality = self._evaluate_response_quality(reasoning_chain.final_answer, test)
            
            # Test passes if confidence in range and response quality is acceptable
            passed = confidence_in_range and response_quality >= 0.6
            
            return BenchmarkResult(
                test_name=test.name,
                passed=passed,
                actual_confidence=confidence,
                reasoning_time=reasoning_time,
                response_quality=response_quality
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name=test.name,
                passed=False,
                actual_confidence=0.0,
                reasoning_time=0.0,
                response_quality=0.0,
                error_message=str(e)
            )
    
    def _evaluate_response_quality(self, response: str, test: BenchmarkTest) -> float:
        """Evaluate quality of reasoning response"""
        if not response or len(response) < 20:
            return 0.2
        
        quality_score = 0.5  # Base score for having content
        
        # Length factor (responses should be substantive but not overly verbose)
        word_count = len(response.split())
        if 20 <= word_count <= 200:
            quality_score += 0.2
        elif 200 < word_count <= 500:
            quality_score += 0.1
        
        # Structure indicators
        structure_indicators = ['therefore', 'however', 'furthermore', 'conclusion', 'analysis', 'evidence']
        structure_count = sum(1 for indicator in structure_indicators if indicator.lower() in response.lower())
        quality_score += min(structure_count * 0.05, 0.2)
        
        # Content relevance (check for query terms)
        query_terms = set(test.query.lower().split())
        response_terms = set(response.lower().split())
        overlap = len(query_terms.intersection(response_terms))
        quality_score += min(overlap / max(len(query_terms), 1), 0.1)
        
        return min(quality_score, 1.0)
    
    def _calculate_summary(self, total_time: float) -> Dict[str, Any]:
        """Calculate benchmark summary statistics"""
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        # Performance metrics
        response_times = [r.reasoning_time for r in self.results]
        confidences = [r.actual_confidence for r in self.results]
        qualities = [r.response_quality for r in self.results]
        
        summary = {
            "total_tests": len(self.results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "pass_rate": len(passed_tests) / len(self.results) * 100,
            "total_time": total_time,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_confidence": statistics.mean(confidences) if confidences else 0,
            "avg_quality": statistics.mean(qualities) if qualities else 0,
            "test_results": self.results
        }
        
        # Get engine performance report
        if hasattr(self.engine, 'get_performance_report'):
            summary["engine_performance"] = self.engine.get_performance_report()
        
        # Get engine score
        if hasattr(self.engine, 'calculate_reasoning_score'):
            summary["engine_score"] = self.engine.calculate_reasoning_score()
        
        return summary
    
    def _print_results(self, summary: Dict[str, Any]):
        """Print comprehensive benchmark results"""
        print("="*60)
        print("🎯 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        # Test results
        print(f"📊 Test Results: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['pass_rate']:.1f}%)")
        
        # Performance metrics
        print(f"⚡ Performance Metrics:")
        print(f"   Average Response Time: {summary['avg_response_time']:.3f}s")
        print(f"   Median Response Time: {summary['median_response_time']:.3f}s")
        print(f"   Range: {summary['min_response_time']:.3f}s - {summary['max_response_time']:.3f}s")
        print(f"   Total Benchmark Time: {summary['total_time']:.2f}s")
        
        # Quality metrics
        print(f"🎯 Quality Metrics:")
        print(f"   Average Confidence: {summary['avg_confidence']:.3f}")
        print(f"   Average Response Quality: {summary['avg_quality']:.3f}")
        
        # Engine score
        if "engine_score" in summary:
            score = summary["engine_score"]
            print(f"🏆 Engine Score: {score['total_score']}/100 ({score['grade']})")
            
            print(f"📈 Score Breakdown:")
            max_scores = {'confidence_quality': 25, 'response_synthesis': 20, 'semantic_analysis': 20, 'pattern_selection': 15, 'learning_adaptation': 10, 'performance': 10}
            for component, score_value in score['components'].items():
                max_score = max_scores[component]
                percentage = (score_value / max_score) * 100
                status = '🟢' if percentage >= 80 else '🟡' if percentage >= 60 else '🔴'
                print(f"   {status} {component}: {score_value:.1f}/{max_score} ({percentage:.0f}%)")
        
        # Failed tests analysis
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print(f"❌ Failed Tests Analysis:")
            for result in failed_tests:
                print(f"   {result.test_name}: {result.error_message}")
        
        # Engine performance
        if "engine_performance" in summary:
            perf = summary["engine_performance"]
            if isinstance(perf, dict) and "status" not in perf:
                print(f"📊 Engine Performance:")
                print(f"   Queries processed: {perf.get('total_queries', 0)}")
                print(f"   QPS: {perf.get('queries_per_second', 0):.1f}")
                print(f"   Error rate: {perf.get('error_rate', 0):.1%}")
        
        print("="*60)

def main():
    """Main benchmark execution"""
    print("🚀 Advanced Reasoning Engine Benchmark Suite")
    print("="*60)
    
    # Create and run benchmark
    benchmark = ReasoningBenchmark()
    results = benchmark.run_benchmark()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    serializable_results = {
        "timestamp": timestamp,
        "summary": {
            k: v for k, v in results.items() 
            if k != "test_results" and not callable(v)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    # Return final assessment
    if results.get("pass_rate", 0) >= 80:
        print("🎉 BENCHMARK PASSED: Reasoning engine meets performance standards!")
        return 0
    elif results.get("pass_rate", 0) >= 60:
        print("👍 BENCHMARK ACCEPTABLE: Reasoning engine needs minor improvements")
        return 1
    else:
        print("⚠️ BENCHMARK FAILED: Reasoning engine requires significant improvements")
        return 2

if __name__ == "__main__":
    exit(main())