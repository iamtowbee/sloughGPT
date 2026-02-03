#!/usr/bin/env python3
"""
Test Infrastructure Validation
Validate that test infrastructure is properly set up
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test that test directory structure exists"""
    base_path = Path(__file__).parent
    required_dirs = [
        "unit",
        "integration", 
        "performance",
        "security",
        "api"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✅ {dir_name}/ directory exists")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    return True

def test_test_files():
    """Test that test files exist"""
    base_path = Path(__file__).parent
    required_files = [
        "unit/test_core_modules.py",
        "integration/test_end_to_end.py", 
        "performance/test_benchmarks.py",
        "security/test_vulnerability_scanning.py",
        "api/test_endpoints.py",
        "conftest.py",
        "run_tests.py",
        "requirements-test.txt"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = base_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            print(f"✅ {file_name} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def test_pytest_availability():
    """Test that pytest is available"""
    try:
        import pytest
        print(f"✅ pytest version {pytest.__version__} is available")
        return True
    except ImportError:
        print("❌ pytest is not available")
        return False

def test_test_runner():
    """Test that test runner is executable"""
    runner_path = Path(__file__).parent / "run_tests.py"
    if runner_path.exists() and os.access(runner_path, os.X_OK):
        print("✅ run_tests.py is executable")
        return True
    else:
        print("❌ run_tests.py is not executable")
        return False

def main():
    """Main validation function"""
    print("🔍 Validating SloughGPT Test Infrastructure")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Test Files", test_test_files),
        ("Pytest Availability", test_pytest_availability),
        ("Test Runner", test_test_runner)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{test_name:20} : {status}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate == 100:
        print("\n🎉 Test infrastructure is properly configured!")
        return 0
    else:
        print("\n❌ Some test infrastructure components need attention.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)