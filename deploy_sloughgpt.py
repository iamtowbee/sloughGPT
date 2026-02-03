#!/usr/bin/env python3
"""
SloughGPT WebUI Launcher
Modified OpenWebUI deployment script for SloughGPT integration
"""

import os
import sys
import subprocess
import signal
from pathlib import Path

def signal_handler(sig, frame):
    print('\n🛑 Shutting down SloughGPT WebUI...')
    sys.exit(0)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python
    try:
        import sys
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version}")
    except ImportError:
        print("❌ Python not available")
        return False
    
    # Check FastAPI
    try:
        import fastapi
        print("✅ FastAPI available")
    except ImportError:
        print("⚠️  Installing FastAPI...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    
    return True

def setup_environment():
    """Setup environment variables for SloughGPT"""
    os.environ.update({
        'SLOGH_GPT_WEBUI_NAME': 'SloughGPT',
        'WEBUI_NAME': 'SloughGPT',
        'WEBUI_FAVICON_URL': 'https://sloughgpt.com/favicon.png',
        'HOST': '0.0.0.0',
        'PORT': '8080',
        'CORS_ALLOW_ORIGIN': '*',
        'DO_NOT_TRACK': 'true',
        'ANONYMIZED_TELEMETRY': 'false'
    })

def start_backend():
    """Start the backend server"""
    print("🚀 Starting SloughGPT WebUI Backend...")
    
    backend_dir = Path(__file__).parent / "openwebui-source" / "backend"
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Start the server
    try:
        # Try to use uvicorn directly
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "open_webui.main:app",
            "--host", "0.0.0.0", 
            "--port", "8080",
            "--reload"
        ]
        
        print(f"📍 Command: {' '.join(cmd)}")
        print("🌐 Server will be available at: http://0.0.0.0:8080")
        print("🎨 WebUI will be accessible at: http://localhost:8080")
        print("⚙️  Press Ctrl+C to stop the server")
        print()
        
        subprocess.run(cmd, env={**os.environ})
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down SloughGPT WebUI...")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        print("📋 Trying alternative approach...")
        
        # Fallback: try running main.py directly
        try:
            subprocess.run([sys.executable, "-m", "open_webui.main"], env={**os.environ})
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False
    
    return True

def start_simple_server():
    """Start a simple FastAPI server with SloughGPT integration"""
    print("🚀 Starting Simple SloughGPT WebUI...")
    
    # Change to root directory
    os.chdir(Path(__file__).parent)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8080))
    
    try:
        # Import and run enhanced webui
        sys.path.insert(0, str(Path(__file__).parent))
        from enhanced_webui import app
        import uvicorn
        
        print(f"🌐 Server will be available at: http://0.0.0.0:{port}")
        print("⚙️  Press Ctrl+C to stop the server")
        print()
        
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except ImportError:
        print("❌ Enhanced WebUI not available, trying simple version...")
        try:
            from simple_webui import app
            import uvicorn
            
            print("🌐 Server will be available at: http://0.0.0.0:8080")
            print("⚙️  Press Ctrl+C to stop the server")
            print()
            
            uvicorn.run(app, host="0.0.0.0", port=8080)
        except ImportError:
            print("❌ No WebUI components found")
            return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    print("="*60)
    print("🚀 SloughGPT WebUI Deployment")
    print("="*60)
    print()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup environment
    setup_environment()
    
    # Check if we want to use OpenWebUI or simple version
    use_openwebui = "--openwebui" in sys.argv
    
    if use_openwebui:
        print("🎯 Using OpenWebUI backend...")
        
        if not check_dependencies():
            print("❌ Dependencies not met, falling back to simple version")
            return start_simple_server()
        
        return start_backend()
    else:
        print("🎯 Using Simple SloughGPT WebUI...")
        return start_simple_server()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Deployment failed")
        print("📋 Try running with: python3 deploy_sloughgpt.py --openwebui")
        sys.exit(1)
    else:
        print("\n✅ Deployment completed")