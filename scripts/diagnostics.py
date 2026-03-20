#!/usr/bin/env python3
"""
System Diagnostics Script
Comprehensive system health check for SloughGPT.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import platform
import time


def check_python():
    """Check Python environment."""
    print("\n[Python Environment]")
    print(f"  Version: {platform.python_version()}")
    print(f"  Implementation: {platform.python_implementation()}")

    required = (3, 8)
    current = sys.version_info[:2]
    if current >= required:
        print(f"  [OK] Python >= 3.8")
    else:
        print(f"  [FAIL] Python 3.8+ required")


def check_packages():
    """Check required packages."""
    print("\n[Required Packages]")

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("psutil", "psutil"),
    ]

    for module, name in packages:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {name}: {ver}")
        except ImportError:
            print(f"  [MISSING] {name}")


def check_gpu():
    """Check GPU availability."""
    print("\n[GPU Support]")

    try:
        import torch
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")

        print(f"  MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("  [SKIP] PyTorch not installed")


def check_directories():
    """Check required directories."""
    print("\n[Directories]")

    dirs = [
        ("models", "Models"),
        ("datasets", "Datasets"),
        ("experiments", "Experiments"),
    ]

    for path, name in dirs:
        if os.path.isdir(path):
            files = len(list(os.listdir(path)))
            print(f"  [OK] {name}/ ({files} files)")
        else:
            print(f"  [MISSING] {name}/")


def check_environment():
    """Check environment variables."""
    print("\n[Environment Variables]")

    env_vars = [
        ("SLAUGHGPT_API_KEY", "API Key"),
        ("SLAUGHGPT_JWT_SECRET", "JWT Secret"),
        ("MODEL_PATH", "Model Path"),
    ]

    for var, name in env_vars:
        val = os.environ.get(var)
        if val:
            if "SECRET" in var or "KEY" in var:
                print(f"  [SET] {name}: {'*' * 8}...")
            else:
                print(f"  [SET] {name}: {val}")
        else:
            print(f"  [NOT SET] {name}")


def check_api_server():
    """Check if API server is running."""
    print("\n[API Server]")

    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            print(f"  [OK] Server running")
            print(f"    Model loaded: {data.get('model_loaded', False)}")
        else:
            print(f"  [FAIL] Server returned {r.status_code}")
    except:
        print("  [NOT RUNNING] Start with: python3 cli.py serve")


def check_disk_space():
    """Check disk space."""
    print("\n[Disk Space]")

    try:
        import psutil
        disk = psutil.disk_usage("/")
        print(f"  Total: {disk.total / (1024**3):.1f} GB")
        print(f"  Used: {disk.used / (1024**3):.1f} GB ({disk.percent}%)")
        print(f"  Free: {disk.free / (1024**3):.1f} GB")

        if disk.percent > 90:
            print("  [WARN] Disk usage > 90%")
        else:
            print("  [OK] Disk usage normal")
    except:
        print("  [SKIP] psutil not installed")


def main():
    print("=" * 50)
    print("SloughGPT System Diagnostics")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    check_python()
    check_packages()
    check_gpu()
    check_directories()
    check_environment()
    check_api_server()
    check_disk_space()

    print("\n" + "=" * 50)
    print("Diagnostics Complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
