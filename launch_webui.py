#!/usr/bin/env python3
"""
WebUI launcher script with multiple options
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_simple_webui():
    """Run the simple webui"""
    print("🚀 Starting Simple WebUI...")
    subprocess.run([sys.executable, "simple_webui.py"])

def run_enhanced_webui():
    """Run the enhanced webui"""
    print("🚀 Starting Enhanced WebUI...")
    subprocess.run([sys.executable, "enhanced_webui.py"])

def run_cerebro_webui():
    """Run the cerebro webui with proper setup"""
    print("🚀 Starting Cerebro WebUI...")
    cerebro_dir = Path("packages/apps/apps/cerebro")
    
    if not cerebro_dir.exists():
        print("❌ Cerebro directory not found")
        return
    
    os.chdir(cerebro_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)
    env['PORT'] = '8080'
    env['HOST'] = '0.0.0.0'
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "open_webui.main:app",
            "--host", "0.0.0.0", 
            "--port", "8080"
        ], env=env)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Cerebro WebUI...")
    except Exception as e:
        print(f"❌ Error starting Cerebro WebUI: {e}")

def main():
    parser = argparse.ArgumentParser(description="SloughGPT WebUI Launcher")
    parser.add_argument(
        "--mode", 
        choices=["simple", "enhanced", "cerebro"], 
        default="enhanced",
        help="WebUI mode to launch (default: enhanced)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="Port to run on (default: 8080)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['PORT'] = str(args.port)
    os.environ['HOST'] = args.host
    
    print(f"🌐 SloughGPT WebUI Launcher")
    print(f"📍 Mode: {args.mode}")
    print(f"🌍 Host: {args.host}")
    print(f"🚪 Port: {args.port}")
    print()
    
    if args.mode == "simple":
        run_simple_webui()
    elif args.mode == "enhanced":
        run_enhanced_webui()
    elif args.mode == "cerebro":
        run_cerebro_webui()

if __name__ == "__main__":
    main()