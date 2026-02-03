#!/usr/bin/env python3
"""
Docker Containerization Validation Script
Test and validate SloughGPT Docker setup
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_command(cmd, description, timeout=60):
    """Run command with error handling"""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False, str(e)

def check_docker_files():
    """Check if Docker files exist"""
    print("\n🐋 Checking Docker files...")
    
    required_files = [
        "Dockerfile",
        "Dockerfile.dev", 
        "Dockerfile.test",
        "docker-compose.yml",
        ".env.example",
        "docker-manage.sh"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    return len(missing_files) == 0, missing_files

def check_docker_directories():
    """Check if Docker directories exist"""
    print("\n📁 Checking Docker directories...")
    
    required_dirs = [
        "docker/nginx",
        "docker/postgres",
        "docker/ssl",
        "docker/grafana/provisioning"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0, missing_dirs

def check_docker_images():
    """Check if Docker can build images"""
    print("\n🏗️ Testing Docker image builds...")
    
    # Test production image build (dry run)
    cmd = ["docker", "build", "--dry-run", "-f", "Dockerfile", "."]
    success, output = run_command(cmd, "Testing production Dockerfile (dry run)")
    
    if not success:
        # Try without --dry-run if not supported
        cmd = ["docker", "build", "--no-cache", "-t", "sloughgpt:test", "-f", "Dockerfile.test", "."]
        success, output = run_command(cmd, "Building test Docker image")
    
    return success

def check_docker_compose():
    """Check Docker Compose configuration"""
    print("\n🔧 Checking Docker Compose configuration...")
    
    cmd = ["docker-compose", "-f", "docker-compose.yml", "config"]
    success, output = run_command(cmd, "Validating docker-compose.yml")
    
    if success:
        print("✅ Docker Compose configuration is valid")
        
        # Parse and show services
        try:
            config_cmd = ["docker-compose", "-f", "docker-compose.yml", "config", "--services"]
            success, services = run_command(config_cmd, "Listing services")
            if success:
                service_list = services.strip().split('\n')
                print(f"📋 Services found: {len(service_list)}")
                for service in service_list:
                    if service.strip():
                        print(f"  - {service.strip()}")
        except Exception as e:
            print(f"⚠️ Could not list services: {str(e)}")
    
    return success

def check_environment_file():
    """Check environment configuration"""
    print("\n🔧 Checking environment configuration...")
    
    env_file = Path(".env.example")
    if not env_file.exists():
        print("❌ .env.example not found")
        return False
    
    # Read and validate environment variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    required_vars = [
        "SLAUGHGPT_ENV",
        "SLAUGHGPT_LOG_LEVEL",
        "DATABASE_URL", 
        "REDIS_URL",
        "SLAUGHGPT_PORT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if f"{var}=" in env_content:
            print(f"✅ {var}")
        else:
            print(f"❌ {var} - MISSING")
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def check_docker_management_script():
    """Check Docker management script"""
    print("\n📜 Checking Docker management script...")
    
    script_path = Path("docker-manage.sh")
    if not script_path.exists():
        print("❌ docker-manage.sh not found")
        return False
    
    # Check if script is executable
    if os.access(script_path, os.X_OK):
        print("✅ docker-manage.sh is executable")
        
        # Test help command
        cmd = ["./docker-manage.sh", "help"]
        success, output = run_command(cmd, "Testing docker-manage.sh help")
        return success
    else:
        print("❌ docker-manage.sh is not executable")
        return False

def check_docker_daemon():
    """Check if Docker daemon is running"""
    print("\n🐋 Checking Docker daemon...")
    
    cmd = ["docker", "version"]
    success, output = run_command(cmd, "Checking Docker version")
    
    if success:
        print("✅ Docker daemon is running")
        # Extract version info
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if 'Version:' in line:
                    print(f"📋 {line.strip()}")
        except:
            pass
    else:
        print("❌ Docker daemon is not running")
    
    return success

def test_docker_build_simple():
    """Test a simple Docker build"""
    print("\n🏗️ Testing simple Docker build...")
    
    # Create a minimal test Dockerfile
    test_dockerfile = """
FROM python:3.10-slim
RUN echo "Docker build test successful"
"""
    
    with open("Dockerfile.test.minimal", "w") as f:
        f.write(test_dockerfile)
    
    try:
        cmd = ["docker", "build", "-t", "sloughgpt:test-minimal", "-f", "Dockerfile.test.minimal", "."]
        success, output = run_command(cmd, "Building minimal test image")
        
        # Cleanup
        os.remove("Dockerfile.test.minimal")
        if success:
            subprocess.run(["docker", "rmi", "sloughgpt:test-minimal"], capture_output=True)
        
        return success
    except Exception as e:
        print(f"❌ Build test failed: {str(e)}")
        return False

def generate_summary(results):
    """Generate test summary"""
    print("\n" + "="*60)
    print("📊 DOCKER VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} : {status}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate == 100:
        print("\n🎉 Docker containerization is properly configured!")
        print("\n📝 Next Steps:")
        print("1. Copy .env.example to .env and configure your settings")
        print("2. Run './docker-manage.sh build' to build Docker images")
        print("3. Run './docker-manage.sh start' to start production services")
        print("4. Run './docker-manage.sh dev' to start development services")
        print("5. Run './docker-manage.sh test' to run tests in Docker")
        
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} Docker configuration issues need to be resolved.")
        return 1

def main():
    """Main validation function"""
    print("🐋 SloughGPT Docker Validation")
    print("="*60)
    
    # Run all validation checks
    results = {}
    
    # Basic checks
    results["Docker Files"] = check_docker_files()[0]
    results["Docker Directories"] = check_docker_directories()[0]
    results["Docker Daemon"] = check_docker_daemon()
    results["Docker Compose"] = check_docker_compose()
    results["Environment File"] = check_environment_file()[0]
    results["Management Script"] = check_docker_management_script()
    
    # Build tests (only if Docker is available)
    if results.get("Docker Daemon", False):
        results["Simple Build Test"] = test_docker_build_simple()
    
    return generate_summary(results)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)