#!/usr/bin/env python3
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

@app.get("/", response_class=HTMLResponse)
async def production_root():
    """Production landing page"""
    html_content = "<!DOCTYPE html><html><head><title>SloughGPT - Production</title><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><link rel=\"icon\" href=\"https://sloughgpt.com/favicon.png\"><meta http-equiv=\"refresh\" content=\"0; url=/docs\"></head><body><div style=\"display: flex; justify-content: center; align-items: center; height: 100vh; font-family: system-ui;\"><div><h1>🚀 SloughGPT Production</h1><p>Redirecting to API documentation...</p></div></div></body></html>"
    return html_content

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
