#!/usr/bin/env python3
"""
SloughGPT Production API Server
FastAPI-based REST API for model inference and management
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import psutil
import os

# Import SloughGPT modules
from .config import ModelConfig
from .neural_network import SloughGPT
from .core.exceptions import SloughGPTError, create_error

# Optimizations - optional import
try:
    from optimizations import OptimizedSloughGPT, create_optimized_model
except ImportError:
    OptimizedSloughGPT = None
    create_optimized_model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_instance = None
model_config = None
server_stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens_generated": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting SloughGPT API Server...")
    await initialize_model()
    logger.info("✅ SloughGPT API Server ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SloughGPT API Server...")

# Create FastAPI app
app = FastAPI(
    title="SloughGPT API",
    description="Production API for SloughGPT neural network",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    input_text: str = Field(..., min_length=1, max_length=4096, description="Input text for generation")
    max_length: int = Field(50, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    use_cache: bool = Field(True, description="Whether to use KV cache")

class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: float
    model_info: Dict[str, Any]

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    total_parameters: int
    trainable_parameters: int
    device: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    model_size_mb: float
    optimization_status: Dict[str, bool]

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    requests_processed: int
    success_rate: float
    model_loaded: bool

class TokenizeRequest(BaseModel):
    """Request model for tokenization"""
    text: str = Field(..., min_length=1, description="Text to tokenize")

class TokenizeResponse(BaseModel):
    """Response model for tokenization"""
    tokens: List[int]
    token_count: int
    detokenized_text: str

# Dependency to get model instance
async def get_model():
    """Get the model instance"""
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance

# Model initialization
async def initialize_model():
    """Initialize the model on startup"""
    global model_instance, model_config
    
    try:
        # Load configuration
        model_config = ModelConfig()
        logger.info(f"📋 Model configuration: {model_config.to_dict()}")
        
        # Create optimized model for production
        if create_optimized_model is not None:
            model_instance = create_optimized_model(
                model_config,
                enable_quantization=True,
                enable_mixed_precision=True
            )
        else:
            # Fallback to standard model
            model_instance = SloughGPT(model_config)
        
        # Move to evaluation mode
        model_instance.eval()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Parameters: {model_instance.count_parameters():,}")
        logger.info(f"   Device: {model_instance.device}")
        logger.info(f"   Optimizations: {model_instance.get_optimization_summary()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic info"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SloughGPT API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }
            code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            .status { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 SloughGPT API Server</h1>
            <p class="status">✅ Server is running and model is loaded!</p>
            
            <h2>📚 Available Endpoints</h2>
            
            <div class="endpoint">
                <h3>🏠 Health Check</h3>
                <p><code>GET /health</code></p>
                <p>Check server health and statistics</p>
            </div>
            
            <div class="endpoint">
                <h3>🤖 Model Info</h3>
                <p><code>GET /model/info</code></p>
                <p>Get detailed model information</p>
            </div>
            
            <div class="endpoint">
                <h3>📝 Text Generation</h3>
                <p><code>POST /generate</code></p>
                <p>Generate text using the model</p>
            </div>
            
            <div class="endpoint">
                <h3>🔤 Tokenization</h3>
                <p><code>POST /tokenize</code></p>
                <p>Tokenize/detokenize text</p>
            </div>
            
            <div class="endpoint">
                <h3>📊 Server Stats</h3>
                <p><code>GET /stats</code></p>
                <p>Get server performance statistics</p>
            </div>
            
            <h2>📖 API Documentation</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            <p>Visit <a href="/redoc">/redoc</a> for ReDoc documentation</p>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    global server_stats
    
    uptime = time.time() - server_stats["start_time"]
    memory_usage = psutil.virtual_memory().used / 1024**2  # MB
    cpu_usage = psutil.cpu_percent()
    
    success_rate = (
        server_stats["successful_requests"] / server_stats["total_requests"] * 100
        if server_stats["total_requests"] > 0 else 100.0
    )
    
    return HealthStatus(
        status="healthy",
        uptime_seconds=uptime,
        memory_usage_mb=memory_usage,
        cpu_usage_percent=cpu_usage,
        requests_processed=server_stats["total_requests"],
        success_rate=success_rate,
        model_loaded=model_instance is not None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = model_instance.get_model_info()
    
    # Get optimization summary if available
    optimization_status = {}
    if hasattr(model_instance, 'get_optimization_summary'):
        summary = model_instance.get_optimization_summary()
        optimization_status = {
            'quantization': summary.get('quantization_enabled', False),
            'compilation': summary.get('compilation_enabled', False),
            'mixed_precision': summary.get('optimizations', {}).get('mixed_precision', False)
        }
    
    return ModelInfo(
        model_type=model_info.get('model_type', 'SloughGPT'),
        total_parameters=model_info.get('total_parameters', 0),
        trainable_parameters=model_info.get('trainable_parameters', 0),
        device=model_info.get('device', 'unknown'),
        vocab_size=model_info.get('vocab_size', 0),
        hidden_size=model_info.get('hidden_size', 0),
        num_layers=model_info.get('num_layers', 0),
        num_heads=model_info.get('num_attention_heads', 0),
        model_size_mb=model_info.get('model_size_mb', 0),
        optimization_status=optimization_status
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    model = Depends(get_model),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate text using the model"""
    global server_stats
    
    try:
        # Update stats
        server_stats["total_requests"] += 1
        
        # Tokenize input (simple character-level for now)
        # In a real implementation, you'd use proper tokenization
        input_tokens = [ord(c) % model.config.vocab_size for c in request.input_text]
        input_tensor = torch.tensor([input_tokens], dtype=torch.long)
        
        # Generate text
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_tensor,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
        
        generation_time = (time.time() - start_time) * 1000  # ms
        
        # Convert generated IDs back to text (simple character-level)
        generated_tokens = generated_ids[0][len(input_tokens):].tolist()
        generated_text = ''.join([chr(token % 256) for token in generated_tokens])
        
        # Update stats
        server_stats["successful_requests"] += 1
        server_stats["total_tokens_generated"] += len(generated_tokens)
        
        # Background task for logging
        background_tasks.add_task(
            log_generation_request,
            request.input_text,
            generated_text,
            generation_time
        )
        
        model_info = model.get_model_info()
        
        return GenerationResponse(
            generated_text=generated_text,
            input_tokens=len(input_tokens),
            output_tokens=len(generated_tokens),
            generation_time_ms=generation_time,
            model_info=model_info
        )
        
    except Exception as e:
        server_stats["failed_requests"] += 1
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest):
    """Tokenize and detokenize text"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Simple character-level tokenization
    # In a real implementation, you'd use proper tokenization
    tokens = [ord(c) % model_instance.config.vocab_size for c in request.text]
    
    # Detokenize
    detokenized = ''.join([chr(token % 256) for token in tokens])
    
    return TokenizeResponse(
        tokens=tokens,
        token_count=len(tokens),
        detokenized_text=detokenized
    )

@app.get("/stats")
async def get_server_stats():
    """Get server performance statistics"""
    global server_stats
    
    uptime = time.time() - server_stats["start_time"]
    
    return {
        "uptime_seconds": uptime,
        "total_requests": server_stats["total_requests"],
        "successful_requests": server_stats["successful_requests"],
        "failed_requests": server_stats["failed_requests"],
        "total_tokens_generated": server_stats["total_tokens_generated"],
        "average_generation_time_ms": (
            server_stats["total_tokens_generated"] / server_stats["successful_requests"]
            if server_stats["successful_requests"] > 0 else 0
        ),
        "requests_per_second": server_stats["total_requests"] / uptime if uptime > 0 else 0
    }

# Background task for logging
async def log_generation_request(input_text: str, generated_text: str, generation_time: float):
    """Log generation request in background"""
    logger.info(f"📝 Generated {len(generated_text)} chars in {generation_time:.1f}ms")

# Error handlers
@app.exception_handler(SloughGPTError)
async def sloughgpt_exception_handler(request, exc: SloughGPTError):
    """Handle SloughGPT-specific exceptions"""
    logger.error(f"SloughGPT Error: {exc.to_json()}")
    return JSONResponse(
        status_code=500,
        content={"error": exc.to_dict()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "details": str(exc)}}
    )

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    return app

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )