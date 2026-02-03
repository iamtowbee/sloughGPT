from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import asyncio

from fastapi import HTTPException, Depends
from fastapi import HTTPException, Depends
from ..core.model_manager import ModelManager
from ..core.cache_manager import CacheManager
from ..core.async_dataset_manager import AsyncDatasetManager
from ..dependencies import get_model_manager, get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input prompt for generation")
    max_tokens: Optional[int] = Field(1024, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: Optional[bool] = Field(False, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated text response")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    model: str = Field(..., description="Model name used")
    timestamp: float = Field(..., description="Response timestamp")

class StreamChatResponse(BaseModel):
    token: str = Field(..., description="Generated token")
    finished: bool = Field(False, description="Whether generation is complete")

@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Generate text completion with caching and optimization"""
    
    start_time = time.time()
    
    try:
        # Validate prompt length
        if len(request.prompt) > 4096:
            raise HTTPException(status_code=400, detail="Prompt too long")
            
        # Generate response
        response_text = await model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate usage (simplified)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(response_text.split())
        
        # Log usage in background
        background_tasks.add_task(
            log_usage,
            prompt_tokens,
            completion_tokens,
            time.time() - start_time
        )
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            model="slogpt",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

@router.post("/completions/stream")
async def chat_completion_stream(
    request: ChatRequest,
    model_manager: ModelManager = Depends(lambda: model_manager)
):
    """Generate streaming text completion"""
    
    if not request.stream:
        raise HTTPException(status_code=400, detail="Streaming must be enabled for this endpoint")
    
    async def generate_stream():
        try:
            # For now, generate all at once and stream tokens
            # In production, you'd implement true token-by-token streaming
            full_response = await model_manager.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Stream word by word
            words = full_response.split()
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield f"data: {StreamChatResponse(token=token, finished=False).json()}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
                
            # Send final message
            yield f"data: {StreamChatResponse(token='', finished=True).json()}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {{'error': 'Generation failed'}}\n\n"
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "slogpt",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "slogpt"
            }
        ]
    }

@router.get("/usage")
async def get_usage_stats(
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Get usage statistics"""
    try:
        cache_stats = await cache_manager.get_stats()
        return {
            "usage": {
                "total_requests": 0,  # You'd implement this with a proper metrics system
                "cache_hit_rate": 0.0,  # Calculate from cache stats
                "average_response_time": 0.0
            },
            "cache": cache_stats
        }
    except Exception as e:
        logger.error(f"Usage stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage stats")

async def log_usage(prompt_tokens: int, completion_tokens: int, response_time: float):
    """Log usage metrics (would integrate with your monitoring system)"""
    logger.info(f"Usage: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, "
               f"{response_time:.3f}s response time")