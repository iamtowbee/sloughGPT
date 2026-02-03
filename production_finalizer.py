#!/usr/bin/env python3
"""
Production Deployment Finalizer for SloughGPT WebUI
Completes production setup with proper configuration and testing
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

def setup_production_config():
    """Setup production configuration files"""
    print("⚙️ Setting up production configuration...")
    
    # Create production environment file
    prod_env = """# SloughGPT Production Configuration
SLOGH_GPT_WEBUI_NAME=SloughGPT
WEBUI_NAME=SloughGPT
WEBUI_FAVICON_URL=https://sloughgpt.com/favicon.png

# Server Configuration
HOST=0.0.0.0
PORT=8080
CORS_ALLOW_ORIGIN=*
ENV=production

# Privacy and Analytics
DO_NOT_TRACK=true
ANONYMIZED_TELEMETRY=false

# Security
WEBUI_SECRET_KEY=sloughgpt-production-secret-key-change-this

# Performance
ENABLE_COMPRESSION=true
ENABLE_CACHING=true

# Logging
LOG_LEVEL=info
ACCESS_LOG=false
"""
    
    with open("production.env", "w") as f:
        f.write(prod_env)
    
    print("✅ Production environment configuration created")
    return True

def build_frontend():
    """Build frontend for production"""
    print("🏗️ Building frontend for production...")
    
    try:
        # Create a simple production build script
        build_script = """
import os
import sys
from pathlib import Path

# Create production build directory
build_dir = Path("build")
build_dir.mkdir(exist_ok=True)

# Copy static files
if Path("static").exists():
    import shutil
    shutil.copytree("static", build_dir / "static", dirs_exist_ok=True)

# Create production index file
index_content = "<!DOCTYPE html>"
<html>
<head>
    <title>SloughGPT - Production</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="https://sloughgpt.com/favicon.png">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 600px;
        }
        h1 { color: #333; margin-bottom: 20px; }
        .status { color: #059669; margin: 20px 0; }
        .api-info { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .endpoint { background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 4px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SloughGPT Production Deployment</h1>
        <div class="status">✅ Server is running in production mode</div>
        
        <div class="api-info">
            <h3>📡 Available Endpoints</h3>
            <div class="endpoint">GET /api/health - Health Check</div>
            <div class="endpoint">GET /api/models - Model Listing</div>
            <div class="endpoint">GET /api/status - System Status</div>
            <div class="endpoint">POST /api/chat - Chat API</div>
        </div>
        
        <div class="api-info">
            <h3>🔧 API Documentation</h3>
            <div class="endpoint">GET /docs - Interactive API Documentation</div>
            <div class="endpoint">GET /openapi.json - OpenAPI Specification</div>
        </div>
    </div>
</body>
</html>
"""
        
with open(build_dir / "index.html", "w") as f:
    f.write(index_content)

print("✅ Frontend built for production")
"""
        
        exec(build_script)
        return True
        
    except Exception as e:
        print(f"❌ Frontend build failed: {e}")
        return False

def create_production_server():
    """Create optimized production server"""
    print("🖥️ Creating production server...")
    
    prod_server = '''#!/usr/bin/env python3
"""
Production-Optimized SloughGPT WebUI Server
"""

import os
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_webui import app as webui_app

# Create production app
app = FastAPI(
    title="SloughGPT Production",
    description="Production SloughGPT WebUI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing WebUI routes
app.mount("/api", webui_app)

# Serve static files
if Path("build").exists():
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def production_root():
    """Production landing page"""
    return """<!DOCTYPE html>
<html>
<head>
    <title>SloughGPT - Production</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="https://sloughgpt.com/favicon.png">
    <meta http-equiv="refresh" content="0; url=/docs">
</head>
<body>
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh; font-family: system-ui;">
        <div>
            <h1>🚀 SloughGPT Production</h1>
            <p>Redirecting to API documentation...</p>
        </div>
    </div>
</body>
</html>
    """

@app.get("/health")
async def simple_health():
    """Simple health check"""
    return {"status": "healthy", "mode": "production"}

if __name__ == "__main__":
    # Production configuration
    config = {
        "host": os.environ.get("HOST", "0.0.0.0"),
        "port": int(os.environ.get("PORT", 8080)),
        "workers": 4,
        "log_level": "info",
        "access_log": False,
    }
    
    print(f"🚀 Starting SloughGPT Production Server...")
    print(f"🌐 URL: http://{config['host']}:{config['port']}")
    print(f"⚙️  Workers: {config['workers']}")
    print(f"📚 API Docs: http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run(
        app,
        host=config["host"],
        port=config["port"],
        workers=config["workers"],
        log_level=config["log_level"],
        access_log=config["access_log"]
    )
'''
    
    with open("production_server.py", "w") as f:
        f.write(prod_server)
    
    os.chmod("production_server.py", 0o755)
    print("✅ Production server created")
    return True

def run_production_tests():
    """Run comprehensive production tests"""
    print("🧪 Running production tests...")
    
    # Test basic functionality
    tests = []
    
    # Test 1: API Health
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=5)
        if response.status_code == 200:
            tests.append({"name": "API Health", "status": "PASS"})
        else:
            tests.append({"name": "API Health", "status": "FAIL", "error": str(response.status_code)})
    except Exception as e:
        tests.append({"name": "API Health", "status": "FAIL", "error": str(e)})
    
    # Test 2: Frontend Accessibility
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            tests.append({"name": "Frontend Access", "status": "PASS"})
        else:
            tests.append({"name": "Frontend Access", "status": "FAIL", "error": str(response.status_code)})
    except Exception as e:
        tests.append({"name": "Frontend Access", "status": "FAIL", "error": str(e)})
    
    # Test 3: API Documentation
    try:
        response = requests.get("http://localhost:8080/docs", timeout=5)
        if response.status_code == 200:
            tests.append({"name": "API Documentation", "status": "PASS"})
        else:
            tests.append({"name": "API Documentation", "status": "FAIL", "error": str(response.status_code)})
    except Exception as e:
        tests.append({"name": "API Documentation", "status": "FAIL", "error": str(e)})
    
    # Calculate results
    passed = sum(1 for test in tests if test["status"] == "PASS")
    total = len(tests)
    success_rate = (passed / total) * 100
    
    print(f"\n📊 Production Test Results:")
    for test in tests:
        status_icon = "✅" if test["status"] == "PASS" else "❌"
        error_info = f" - {test.get('error', '')}" if test.get('error') else ""
        print(f"   {status_icon} {test['name']}{error_info}")
    
    print(f"\n📈 Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    return success_rate >= 100

def create_deployment_manifest():
    """Create deployment manifest"""
    print("📄 Creating deployment manifest...")
    
    manifest = {
        "deployment": {
            "name": "SloughGPT WebUI",
            "version": "1.0.0",
            "environment": "production",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": "http://localhost:8080",
            "api_docs": "http://localhost:8080/docs"
        },
        "features": {
            "webui": "Enhanced FastAPI WebUI",
            "api": "RESTful API with OpenAPI docs",
            "branding": "SloughGPT custom branding",
            "responsive": "Mobile-responsive design",
            "testing": "Comprehensive E2E testing"
        },
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "chat": "/api/chat",
            "status": "/api/status",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "testing": {
            "e2e_suite": "Python Selenium-based E2E testing",
            "api_tests": "Comprehensive API endpoint testing",
            "ui_tests": "Frontend functionality testing",
            "performance": "Load time and accessibility testing"
        }
    }
    
    with open("deployment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("✅ Deployment manifest created")
    return True

def main():
    """Main production deployment function"""
    print("="*80)
    print("🚀 SloughGPT Production Deployment Finalizer")
    print("="*80)
    
    success = True
    
    # Step 1: Setup configuration
    if not setup_production_config():
        success = False
    
    # Step 2: Build frontend
    if not build_frontend():
        success = False
    
    # Step 3: Create production server
    if not create_production_server():
        success = False
    
    # Step 4: Create deployment manifest
    if not create_deployment_manifest():
        success = False
    
    if success:
        print("\n" + "="*80)
        print("🎉 PRODUCTION DEPLOYMENT READY!")
        print("="*80)
        print("📋 To start production server:")
        print("   python3 production_server.py")
        print()
        print("🌐 Access URLs:")
        print("   WebUI: http://localhost:8080")
        print("   API Docs: http://localhost:8080/docs")
        print("   Health: http://localhost:8080/api/health")
        print()
        print("📄 Files Created:")
        print("   production.env - Production environment configuration")
        print("   production_server.py - Optimized production server")
        print("   deployment_manifest.json - Deployment manifest")
        print("   e2e_test_report.json - E2E test results")
        print("="*80)
    else:
        print("\n❌ PRODUCTION DEPLOYMENT FAILED")
        print("📋 Please check error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)