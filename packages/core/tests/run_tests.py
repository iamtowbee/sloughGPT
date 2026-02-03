#!/usr/bin/env python3
"""
SloughGPT Test Runner
Comprehensive test execution script
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_command(cmd, description, timeout=300):
    """Run a command with error handling and timing"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        print(f"\nDuration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"\nERROR: Command timed out after {timeout} seconds")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False, "", str(e)

def run_unit_tests():
    """Run unit tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "unit/",
        "-v",
        "--tb=short",
        "--cov=sloughgpt",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/unit",
        "--cov-fail-under=70",
        "-x"
    ]
    
    success, stdout, stderr = run_command(cmd, "Unit Tests")
    return success

def run_integration_tests():
    """Run integration tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "integration/",
        "-v",
        "--tb=short",
        "--cov=sloughgpt",
        "--cov-append",
        "--cov-report=html:htmlcov/integration",
        "-x"
    ]
    
    success, stdout, stderr = run_command(cmd, "Integration Tests")
    return success

def run_performance_tests():
    """Run performance tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "performance/",
        "-v",
        "--tb=short",
        "-m", "not slow",
        "--durations=0"
    ]
    
    success, stdout, stderr = run_command(cmd, "Performance Tests")
    return success

def run_security_tests():
    """Run security tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "security/",
        "-v",
        "--tb=short",
        "-x"
    ]
    
    success, stdout, stderr = run_command(cmd, "Security Tests")
    return success

def run_api_tests():
    """Run API tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "api/",
        "-v",
        "--tb=short",
        "-x"
    ]
    
    success, stdout, stderr = run_command(cmd, "API Tests")
    return success

def run_all_tests():
    """Run all tests"""
    test_results = {}
    
    # Run tests in order of importance and speed
    test_order = [
        ("unit", run_unit_tests, "Unit Tests"),
        ("security", run_security_tests, "Security Tests"),
        ("api", run_api_tests, "API Tests"),
        ("integration", run_integration_tests, "Integration Tests"),
        ("performance", run_performance_tests, "Performance Tests")
    ]
    
    print(f"\n{'='*80}")
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}")
    
    total_start = time.time()
    
    for test_type, test_func, test_name in test_order:
        print(f"\n{'#'*60}")
        print(f"#{test_name:^58}#")
        print(f"{'#'*60}")
        
        try:
            success = test_func()
            test_results[test_type] = success
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
                # Stop on critical test failures
                if test_type in ["unit", "security"]:
                    print(f"\n🛑 Stopping test suite due to critical failure in {test_name}")
                    break
                    
        except Exception as e:
            print(f"❌ {test_name} ERROR: {str(e)}")
            test_results[test_type] = False
    
    total_duration = time.time() - total_start
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Test Results:")
    
    passed = 0
    total = len(test_results)
    
    for test_type, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_type:15} : {status}")
        if success:
            passed += 1
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    # Coverage report
    if os.path.exists("htmlcov"):
        print(f"\n📊 Coverage reports generated in htmlcov/")
    
    return success_rate == 100.0

def lint_code():
    """Run code linting and formatting checks"""
    lint_commands = [
        (["flake8", "sloughgpt/", "--max-line-length=100", "--ignore=E501,W503"], "Flake8 Linting"),
        (["black", "--check", "sloughgpt/"], "Black Formatting"),
        (["isort", "--check-only", "sloughgpt/"], "Import Sorting"),
        (["mypy", "sloughgpt/", "--ignore-missing-imports"], "Type Checking")
    ]
    
    print(f"\n{'='*60}")
    print("CODE QUALITY CHECKS")
    print(f"{'='*60}")
    
    all_passed = True
    
    for cmd, description in lint_commands:
        try:
            success, stdout, stderr = run_command(cmd, description, timeout=60)
            if not success:
                all_passed = False
                print(f"❌ {description} FAILED")
            else:
                print(f"✅ {description} PASSED")
        except FileNotFoundError:
            print(f"⚠️  {description} SKIPPED (tool not installed)")
    
    return all_passed

def install_dependencies():
    """Install test dependencies"""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "pytest",
        "pytest-cov",
        "pytest-html",
        "pytest-asyncio",
        "pytest-xdist",
        "pytest-mock",
        "flake8",
        "black",
        "isort",
        "mypy",
        "coverage",
        "aiohttp",
        "pytest-timeout"
    ]
    
    success, stdout, stderr = run_command(cmd, "Installing Test Dependencies", timeout=120)
    return success

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="SloughGPT Test Runner")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "security", "api", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run code quality checks before tests"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install test dependencies before running"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Generate coverage reports"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Change to tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    print("🚀 SloughGPT Test Runner")
    print(f"Working Directory: {test_dir}")
    
    # Install dependencies if requested
    if args.install:
        print("\n📦 Installing test dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return 1
    
    # Run linting if requested
    if args.lint:
        print("\n🔍 Running code quality checks...")
        if not lint_code():
            print("❌ Code quality checks failed")
            return 1
    
    # Run tests
    success = False
    
    try:
        if args.type == "all":
            success = run_all_tests()
        elif args.type == "unit":
            success = run_unit_tests()
        elif args.type == "integration":
            success = run_integration_tests()
        elif args.type == "performance":
            success = run_performance_tests()
        elif args.type == "security":
            success = run_security_tests()
        elif args.type == "api":
            success = run_api_tests()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {str(e)}")
        return 1
    
    # Return appropriate exit code
    if success:
        print(f"\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)