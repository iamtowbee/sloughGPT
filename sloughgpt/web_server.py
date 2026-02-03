#!/usr/bin/env python3
"""
SloughGPT Web Server
Serves the web interface and API together
"""

import asyncio
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from sloughgpt.api_server import create_app

def create_web_server() -> FastAPI:
    """Create integrated web server with both API and web interface"""
    app = create_app()
    
    # Add static file serving for web interface
    web_interface_path = os.path.join(os.path.dirname(__file__), "web_interface.html")
    
    @app.get("/", response_class=FileResponse)
    async def web_interface():
        """Serve the main web interface"""
        return FileResponse(web_interface_path)
    
    return app

if __name__ == "__main__":
    # Create web server
    app = create_web_server()
    
    print("🚀 Starting SloughGPT Web Server...")
    print("📱 Web Interface: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 API Endpoint: http://localhost:8000")
    
    # Run server
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )