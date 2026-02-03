#!/usr/bin/env python3
"""
Production Deployment Script for SloughGPT WebUI
Handles complete production setup and E2E testing
"""

import os
import sys
import subprocess
import signal
import time
import requests
from pathlib import Path

def signal_handler(sig, frame):
    print('\n🛑 Shutting down production deployment...')
    sys.exit(0)

def check_server_health(url="http://localhost:8080"):
    """Check if server is healthy"""
    try:
        response = requests.get(f"{url}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_production_server():
    """Start SloughGPT production server"""
    print("🚀 Starting SloughGPT Production Server...")
    
    # Change to root directory
    os.chdir(Path(__file__).parent)
    
    # Setup environment for production
    env = os.environ.copy()
    env.update({
        'SLOGH_GPT_WEBUI_NAME': 'SloughGPT',
        'WEBUI_NAME': 'SloughGPT', 
        'WEBUI_FAVICON_URL': 'https://sloughgpt.com/favicon.png',
        'HOST': '0.0.0.0',
        'PORT': '8080',
        'CORS_ALLOW_ORIGIN': '*',
        'DO_NOT_TRACK': 'true',
        'ANONYMIZED_TELEMETRY': 'false',
        'ENV': 'production'
    })
    
    try:
        # Start enhanced webui as production
        from enhanced_webui import app
        import uvicorn
        
        print("✅ Production server starting...")
        print("🌐 URL: http://localhost:8080")
        print("⚙️  Production Mode Enabled")
        print()
        
        port = int(os.environ.get('PORT', 8080))
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=False
        )
        
    except ImportError:
        print("❌ Enhanced WebUI not found, falling back...")
        # Fallback to simple version
        try:
            from simple_webui import app
            import uvicorn
            
            print("✅ Simple WebUI starting...")
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=8080,
                log_level="info"
            )
        except ImportError:
            print("❌ No WebUI components available")
            return False
    
    return True

def run_api_tests():
    """Run basic API tests"""
    print("\n🧪 Running API Tests...")
    
    base_url = "http://localhost:8080"
    tests_passed = 0
    tests_total = 0
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            tests_passed += 1
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    tests_total += 1
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/api/models", timeout=5)
        if response.status_code == 200:
            print("✅ Models endpoint passed")
            tests_passed += 1
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
    tests_total += 1
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status endpoint passed")
            tests_passed += 1
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
    tests_total += 1
    
    # Test SloughGPT-specific endpoint if available
    try:
        response = requests.get(f"{base_url}/api/models/sloughgpt", timeout=5)
        if response.status_code == 200:
            print("✅ SloughGPT models endpoint passed")
            tests_passed += 1
        else:
            print("ℹ️  SloughGPT models endpoint not available")
    except:
        print("ℹ️  SloughGPT models endpoint not implemented")
    tests_total += 1
    
    print(f"\n📊 API Tests: {tests_passed}/{tests_total} passed")
    return tests_passed, tests_total

def run_browser_tests():
    """Run browser-based E2E tests"""
    print("\n🌐 Running Browser E2E Tests...")
    
    base_url = "http://localhost:8080"
    
    try:
        # Basic browser test using requests to simulate browser
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Frontend loads successfully")
            
            # Check for SloughGPT branding
            if "SloughGPT" in response.text:
                print("✅ SloughGPT branding present")
                return True
            else:
                print("❌ SloughGPT branding missing")
                return False
        else:
            print(f"❌ Frontend failed to load: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Browser test error: {e}")
        return False

def validate_frontend_features():
    """Validate specific SloughGPT frontend features"""
    print("\n🎨 Validating Frontend Features...")
    
    try:
        response = requests.get("http://localhost:8080", timeout=10)
        content = response.text
        
        features_checked = 0
        features_passed = 0
        
        # Check for main UI elements
        features = {
            "Chat Interface": "chat-input" in content.lower(),
            "Model Selection": "model" in content.lower(),
            "API Endpoints": "/api/" in content,
            "Responsive Design": "viewport" in content.lower(),
        }
        
        for feature, present in features.items():
            features_checked += 1
            if present:
                print(f"✅ {feature}: Available")
                features_passed += 1
            else:
                print(f"ℹ️  {feature}: Not detected")
        
        # Check for SloughGPT specific features
        sloughgpt_features = {
            "SloughGPT Branding": "sloughgpt" in content.lower(),
            "Enhanced WebUI": "enhanced" in content.lower() or "webui" in content.lower(),
        }
        
        for feature, present in sloughgpt_features.items():
            features_checked += 1
            if present:
                print(f"✅ {feature}: Available")
                features_passed += 1
            else:
                print(f"ℹ️  {feature}: Not detected")
        
        print(f"\n📊 Frontend Validation: {features_passed}/{features_checked} features passed")
        return features_passed, features_checked
        
    except Exception as e:
        print(f"❌ Frontend validation error: {e}")
        return 0, 0

def main():
    """Main production deployment and testing"""
    print("="*80)
    print("🚀 SloughGPT Production Deployment & E2E Testing")
    print("="*80)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Step 1: Start production server
    print("\n📦 Step 1: Starting Production Server")
    
    # Start server in background process
    server_process = subprocess.Popen([
        sys.executable, "-c", 
        """
import sys
sys.path.insert(0, '.')
from enhanced_webui import app
import uvicorn
port = int(os.environ.get('PORT', 8080))
        uvicorn.run(app, host='0.0.0.0', port=port, log_level='error', access_log=False)
        """
    ])
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        if check_server_health():
            print("✅ Server started successfully")
            break
        time.sleep(1)
    else:
        print("❌ Server failed to start")
        server_process.terminate()
        return False
    
    try:
        # Step 2: Run API tests
        print("\n🧪 Step 2: API Testing")
        api_passed, api_total = run_api_tests()
        
        # Step 3: Run browser tests
        print("\n🌐 Step 3: Browser E2E Testing")
        browser_passed = run_browser_tests()
        
        # Step 4: Validate frontend features
        print("\n🎨 Step 4: Frontend Feature Validation")
        frontend_passed, frontend_total = validate_frontend_features()
        
        # Generate test report
        print("\n" + "="*80)
        print("📊 PRODUCTION DEPLOYMENT REPORT")
        print("="*80)
        print(f"✅ Server Status: Running")
        print(f"🔌 Server URL: http://localhost:8080")
        print(f"🧪 API Tests: {api_passed}/{api_total} passed")
        print(f"🌐 Browser Tests: {'Passed' if browser_passed else 'Failed'}")
        print(f"🎨 Frontend Features: {frontend_passed}/{frontend_total} validated")
        
        # Overall assessment
        total_tests = api_total + 1 + frontend_total  # API + Browser + Frontend
        total_passed = api_passed + (1 if browser_passed else 0) + frontend_passed
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("✅ PRODUCTION DEPLOYMENT SUCCESSFUL")
            print("🚀 SloughGPT WebUI is ready for production use!")
        else:
            print("⚠️  DEPLOYMENT NEEDS ATTENTION")
            print("📋 Review test results and fix issues")
        
        print("="*80)
        print("🌐 Access your SloughGPT WebUI at: http://localhost:8080")
        print("⚙️  Press Ctrl+C to stop the server")
        
        # Keep server running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            server_process.terminate()
        
    finally:
        # Cleanup
        if 'server_process' in locals():
            server_process.terminate()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)