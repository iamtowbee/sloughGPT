#!/usr/bin/env python3
"""
Performance testing and benchmarking for SloughGPT Enhanced WebUI
"""

import os
import sys
import time
import json
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any
import concurrent.futures
import requests
import psutil
import subprocess
from pathlib import Path

class PerformanceBenchmarker:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": base_url,
            "system_info": self.get_system_info(),
            "benchmarks": {}
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    def benchmark_api_endpoints(self) -> Dict[str, Any]:
        """Benchmark API endpoints"""
        endpoints = [
            "/api/health",
            "/api/models", 
            "/api/status",
            "/api/models/sloughgpt",
            "/api/status/sloughgpt"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            url = f"{self.base_url}{endpoint}"
            times = []
            
            # Make 10 requests to get average
            for _ in range(10):
                start_time = time.time()
                try:
                    response = requests.get(url, timeout=5)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        times.append(end_time - start_time)
                except Exception as e:
                    print(f"Error requesting {endpoint}: {e}")
            
            if times:
                results[endpoint] = {
                    "average_response_time": statistics.mean(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "median_response_time": statistics.median(times),
                    "success_rate": len(times) / 10.0
                }
            else:
                results[endpoint] = {
                    "average_response_time": None,
                    "success_rate": 0.0
                }
        
        return results
    
    def benchmark_concurrent_load(self, concurrent_users: int = 10, duration: int = 60) -> Dict[str, Any]:
        """Benchmark concurrent user load"""
        print(f"Starting concurrent load test: {concurrent_users} users for {duration} seconds")
        
        start_time = time.time()
        end_time = start_time + duration
        
        def make_request():
            """Make a single request"""
            request_start = time.time()
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=10)
                request_end = time.time()
                return {
                    "success": response.status_code == 200,
                    "response_time": request_end - request_start,
                    "timestamp": request_start
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - request_start,
                    "timestamp": request_start
                }
        
        def worker():
            """Worker thread for making requests"""
            results = []
            while time.time() < end_time:
                result = make_request()
                results.append(result)
                time.sleep(1)  # 1 request per second per user
            return results
        
        # Start concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Worker error: {e}")
        
        # Analyze results
        successful_requests = [r for r in all_results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        return {
            "concurrent_users": concurrent_users,
            "duration": duration,
            "total_requests": len(all_results),
            "successful_requests": len(successful_requests),
            "success_rate": len(successful_requests) / len(all_results),
            "requests_per_second": len(all_results) / duration,
            "average_response_time": statistics.mean(response_times) if response_times else None,
            "median_response_time": statistics.median(response_times) if response_times else None,
            "p95_response_time": self.percentile(response_times, 0.95) if response_times else None,
            "p99_response_time": self.percentile(response_times, 0.99) if response_times else None
        }
    
    def percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def benchmark_chat_performance(self) -> Dict[str, Any]:
        """Benchmark chat API performance"""
        print("Benchmarking chat performance...")
        
        chat_payloads = [
            {"message": "Hello", "model": "gpt-3.5-turbo"},
            {"message": "How are you?", "model": "gpt-4"},
            {"message": "Tell me about yourself", "model": "claude-3-sonnet"},
            {"message": "What can you do?", "model": "llama-2-7b"}
        ]
        
        results = []
        
        for payload in chat_payloads:
            times = []
            
            for _ in range(5):  # 5 requests per payload
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=10
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        times.append(end_time - start_time)
                except Exception as e:
                    print(f"Chat API error: {e}")
            
            if times:
                results.append({
                    "model": payload["model"],
                    "message_length": len(payload["message"]),
                    "average_response_time": statistics.mean(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times)
                })
        
        return {
            "chat_benchmarks": results,
            "overall_average": statistics.mean([r["average_response_time"] for r in results]) if results else None
        }
    
    def benchmark_resource_usage(self, duration: int = 30) -> Dict[str, Any]:
        """Monitor resource usage during load"""
        print(f"Monitoring resource usage for {duration} seconds...")
        
        cpu_usage = []
        memory_usage = []
        
        start_time = time.time()
        
        # Start a background load
        def background_load():
            while time.time() - start_time < duration:
                try:
                    requests.get(f"{self.base_url}/api/health", timeout=5)
                except:
                    pass
                time.sleep(0.5)
        
        import threading
        load_thread = threading.Thread(target=background_load)
        load_thread.start()
        
        # Monitor resources
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent())
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)
        
        load_thread.join()
        
        return {
            "duration": duration,
            "cpu_usage": {
                "average": statistics.mean(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory_usage": {
                "average": statistics.mean(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            }
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("🚀 Starting SloughGPT Enhanced WebUI Performance Benchmarks")
        print("=" * 60)
        
        # API endpoint benchmarks
        print("📊 Benchmarking API endpoints...")
        self.results["benchmarks"]["api_endpoints"] = self.benchmark_api_endpoints()
        
        # Concurrent load test
        print("🔄 Running concurrent load test...")
        self.results["benchmarks"]["concurrent_load"] = self.benchmark_concurrent_load()
        
        # Chat performance
        print("💬 Benchmarking chat performance...")
        self.results["benchmarks"]["chat_performance"] = self.benchmark_chat_performance()
        
        # Resource usage
        print("📈 Monitoring resource usage...")
        self.results["benchmarks"]["resource_usage"] = self.benchmark_resource_usage()
        
        # Performance score
        self.results["benchmarks"]["performance_score"] = self.calculate_performance_score()
        
        print("=" * 60)
        print("✅ Performance benchmarks completed!")
        
        return self.results
    
    def calculate_performance_score(self) -> Dict[str, Any]:
        """Calculate overall performance score"""
        benchmarks = self.results["benchmarks"]
        score = 0
        max_score = 100
        
        # API response time score (40 points)
        api_times = []
        for endpoint_data in benchmarks["api_endpoints"].values():
            if endpoint_data.get("average_response_time"):
                api_times.append(endpoint_data["average_response_time"])
        
        if api_times:
            avg_api_time = statistics.mean(api_times)
            if avg_api_time < 0.1:  # <100ms
                api_score = 40
            elif avg_api_time < 0.2:  # <200ms
                api_score = 30
            elif avg_api_time < 0.5:  # <500ms
                api_score = 20
            else:
                api_score = 10
            score += api_score
        
        # Success rate score (20 points)
        concurrent = benchmarks["concurrent_load"]
        if concurrent["success_rate"] > 0.95:
            score += 20
        elif concurrent["success_rate"] > 0.90:
            score += 15
        elif concurrent["success_rate"] > 0.80:
            score += 10
        else:
            score += 5
        
        # Resource efficiency score (20 points)
        resource = benchmarks["resource_usage"]
        avg_cpu = resource["cpu_usage"]["average"]
        avg_memory = resource["memory_usage"]["average"]
        
        if avg_cpu < 50 and avg_memory < 50:
            score += 20
        elif avg_cpu < 70 and avg_memory < 70:
            score += 15
        elif avg_cpu < 85 and avg_memory < 85:
            score += 10
        else:
            score += 5
        
        # Throughput score (20 points)
        rps = concurrent["requests_per_second"]
        if rps > 50:
            score += 20
        elif rps > 30:
            score += 15
        elif rps > 20:
            score += 10
        else:
            score += 5
        
        return {
            "overall_score": score,
            "max_score": max_score,
            "grade": self.get_grade(score / max_score),
            "breakdown": {
                "api_response_time": api_times and min(40, 40 - (statistics.mean(api_times) * 400)),
                "success_rate": min(20, concurrent["success_rate"] * 20),
                "resource_efficiency": min(20, (100 - avg_cpu) + (100 - avg_memory) / 10),
                "throughput": min(20, rps * 0.4)
            }
        }
    
    def get_grade(self, percentage: float) -> str:
        """Get letter grade from percentage"""
        if percentage >= 0.9:
            return "A+"
        elif percentage >= 0.85:
            return "A"
        elif percentage >= 0.8:
            return "B+"
        elif percentage >= 0.75:
            return "B"
        elif percentage >= 0.7:
            return "C+"
        elif percentage >= 0.6:
            return "C"
        elif percentage >= 0.5:
            return "D"
        else:
            return "F"
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {filename}")
    
    def print_summary(self):
        """Print performance summary"""
        benchmarks = self.results["benchmarks"]
        score = benchmarks["performance_score"]
        
        print("\n🎯 PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"Overall Score: {score['overall_score']}/{score['max_score']} ({score['overall_score']/score['max_score']*100:.1f}%)")
        print(f"Grade: {score['grade']}")
        print("")
        
        print("📊 API Endpoints:")
        for endpoint, data in benchmarks["api_endpoints"].items():
            avg_time = data.get("average_response_time", 0) * 1000
            print(f"  {endpoint}: {avg_time:.1f}ms (Success: {data.get('success_rate', 0)*100:.1f}%)")
        print("")
        
        print("🔄 Load Test:")
        concurrent = benchmarks["concurrent_load"]
        print(f"  Success Rate: {concurrent['success_rate']*100:.1f}%")
        print(f"  Requests/sec: {concurrent['requests_per_second']:.1f}")
        print(f"  Avg Response: {concurrent.get('average_response_time', 0)*1000:.1f}ms")
        print(f"  P95 Response: {concurrent.get('p95_response_time', 0)*1000:.1f}ms")
        print("")
        
        print("📈 Resource Usage:")
        resource = benchmarks["resource_usage"]
        print(f"  CPU: {resource['cpu_usage']['average']:.1f}% (max: {resource['cpu_usage']['max']:.1f}%)")
        print(f"  Memory: {resource['memory_usage']['average']:.1f}% (max: {resource['memory_usage']['max']:.1f}%)")
        print("")
        
        print("💡 Recommendations:")
        if score['overall_score'] < 80:
            print("  ⚠️  Performance below optimal. Consider:")
            if concurrent.get('average_response_time', 0) > 0.2:
                print("    - Optimizing API response times")
            if resource['cpu_usage']['average'] > 70:
                print("    - Reducing CPU usage")
            if concurrent['success_rate'] < 0.95:
                print("    - Improving error handling")
        else:
            print("  ✅ Performance is excellent!")

def main():
    """Main function"""
    base_url = os.getenv("BASE_URL", "http://localhost:8080")
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding correctly at {base_url}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server at {base_url}: {e}")
        print("🚀 Please start the server first:")
        print("   python3 enhanced_webui.py")
        return False
    
    # Run benchmarks
    benchmarker = PerformanceBenchmarker(base_url)
    results = benchmarker.run_all_benchmarks()
    
    # Save and display results
    benchmarker.save_results()
    benchmarker.print_summary()
    
    # Exit with success if performance is acceptable
    score = results["benchmarks"]["performance_score"]["overall_score"]
    return score >= 70  # Acceptable performance threshold

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)