from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import time
from typing import Dict, Any
import uvicorn

# TODO: Fix middleware imports - temporarily disabled
# from .middleware import (
#     RequestLoggingMiddleware, 
#     ErrorHandlingMiddleware, 
#     RateLimitMiddleware, 
#     SecurityHeadersMiddleware,
#     AuthenticationMiddleware
# )
from .routes import chat, model, dataset, health, monitoring
from .core.config import settings
from .core.model_manager import ModelManager
from .core.cache_manager import CacheManager
from .core.async_dataset_manager import AsyncDatasetManager
from .monitoring.metrics_collector import metrics_collector
from .dependencies import set_managers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
model_manager = None
cache_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model_manager, cache_manager
    
    logger.info("Starting up SloGPT API Server...")
    
    try:
        # Initialize managers
        cache_manager = CacheManager(settings.REDIS_URL)
        await cache_manager.connect()
        
        model_manager = ModelManager(settings.MODEL_PATH, cache_manager)
        await model_manager.load_model()
        
        # Set global managers for dependency injection
        set_managers(model_manager, cache_manager)
        
        # Initialize async dataset manager
        await dataset.async_dataset_manager.initialize()
        
        # Start metrics collection
        if settings.ENABLE_METRICS:
            await metrics_collector.start_collection(interval=5.0)
        
        # TODO: Fix middleware instantiation - temporarily disabled
        # from .middleware import RateLimitMiddleware
        # rate_limit_middleware = RateLimitMiddleware(app)
        # await rate_limit_middleware.start_cleanup_task()
        
        logger.info("SloGPT API Server started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        logger.info("Shutting down SloGPT API Server...")
        if model_manager:
            await model_manager.cleanup()
        if cache_manager:
            await cache_manager.disconnect()
        
        # Cleanup async dataset manager
        await dataset.async_dataset_manager.cleanup()
        
        # Stop metrics collection
        if settings.ENABLE_METRICS:
            await metrics_collector.stop_collection()

# Create FastAPI app
app = FastAPI(
    title="SloGPT API",
    description="High-performance API for SloGPT model inference",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None
)

# Add middleware in correct order
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware (FastAPI automatically instantiates the classes)
# TODO: Fix middleware instantiation - temporarily disabled for basic functionality
# if settings.ENABLE_AUTH:
#     app.add_middleware(AuthenticationMiddleware, optional=False)
# 
# app.add_middleware(RateLimitMiddleware)
# app.add_middleware(SecurityHeadersMiddleware)
# app.add_middleware(ErrorHandlingMiddleware)
# app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(model.router, prefix="/api/v1/model", tags=["Model"])
app.include_router(dataset.router, prefix="/api/v1/dataset", tags=["Dataset"])
app.include_router(monitoring.router, prefix="/api/v1/metrics", tags=["Monitoring"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SloGPT API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "chat": "/api/v1/chat",
            "model": "/api/v1/model",
            "dataset": "/api/v1/dataset"
        }
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": time.time()
            }
        }
    )



if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )