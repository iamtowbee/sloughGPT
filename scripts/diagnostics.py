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
    print(f"  Compiler: {platform.python_compiler()}")

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
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

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
        ("logs", "Logs"),
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
        ("CUDA_VISIBLE_DEVICES", "CUDA Devices"),
    ]

    for var, name in env_vars:
        val = os.environ.get(var)
        if val:
            if "SECRET" in var or "KEY" in var:
                print(f"  [SET] {name}: {'*' * len(val[:8])}...")
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
            print(f"    Model type: {data.get('model_type', 'N/A')}")
        else:
            print(f"  [FAIL] Server returned {r.status_code}")
    except requests.exceptions.ConnectionError:
        print("  [NOT RUNNING] Start with: python3 cli.py serve")
    except Exception as e:
        print(f"  [ERROR] {e}")


def check_docker():
    """Check Docker status."""
    print("\n[Docker]")

    try:
        import subprocess
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  [OK] {result.stdout.strip()}")

            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5
            )
            containers = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if containers:
                print(f"  Running containers: {len(containers)}")
                for c in containers[:5]:
                    print(f"    - {c}")
            else:
                print("  No running containers")
        else:
            print("  [FAIL] Docker not working")

    except FileNotFoundError:
        print("  [NOT INSTALLED] Docker not found")
    except Exception as e:
        print(f"  [ERROR] {e}")


def check_kubernetes():
    """Check Kubernetes status."""
    print("\n[Kubernetes]")

    try:
        import subprocess
        result = subprocess.run(
            ["kubectl", "version", "--client"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("  [OK] kubectl installed")

            result = subprocess.run(
                ["kubectl", "cluster-info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("  [OK] Cluster connected")
            else:
                print("  [WARN] Cluster not connected")
        else:
            print("  [NOT INSTALLED] kubectl not found")

    except FileNotFoundError:
        print("  [NOT INSTALLED] kubectl not found")
    except Exception as e:
        print(f"  [ERROR] {e}")


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
        elif disk.percent > 80:
            print("  [CAUTION] Disk usage > 80%")
        else:
            print("  [OK] Disk usage normal")

    except ImportError:
        print("  [SKIP] psutil not installed")


def check_memory():
    """Check memory usage."""
    print("\n[Memory]")

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  Total: {mem.total / (1024**3):.1f} GB")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        print(f"  Used: {mem.percent}%")

        if mem.percent > 90:
            print("  [WARN] Memory usage > 90%")
        else:
            print("  [OK] Memory usage normal")

    except ImportError:
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
    check_docker()
    check_kubernetes()
    check_disk_space()
    check_memory()

    print("\n" + "=" * 50)
    print("Diagnostics Complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
