"""Cerebro - Advanced AI Interface with OpenWebUI Backend"""

import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get current directory
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

def main():
    """Main entry point for Cerebro AI Interface"""
    logger.info("🧠 Starting Cerebro - Advanced AI Interface...")
    logger.info("📋 Powered by OpenWebUI backend with SLO integration")
    
    # Set environment variables for OpenWebUI
    os.environ["WEBUI_NAME"] = "Cerebro"
    os.environ["WEBUI_SECRET_KEY"] = os.getenv("WEBUI_SECRET_KEY", "cerebro-secret-key-change-in-production")
    
    try:
        # Import OpenWebUI main app
        import open_webui.main as openwebui_main
        
        # Get the OpenWebUI app
        app = openwebui_main.app
        
        # Customize app branding
        logger.info("🎨 Cerebro AI Interface initializing...")
        logger.info("🌐 Web interface will be available at: http://localhost:8080")
        logger.info("🧠 SLO model integration enabled")
        
        # Start using OpenWebUI's startup
        from openwebui_main import start_webui
        start_webui()
        
    except Exception as e:
        logger.error(f"💥 Failed to start Cerebro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()