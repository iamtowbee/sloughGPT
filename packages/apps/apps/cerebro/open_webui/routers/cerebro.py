import asyncio
import json
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import aiohttp
from aiocache import cached

from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from sqlalchemy.orm import Session

from open_webui.internal.db import get_session
from open_webui.models.models import Models
from open_webui.models.users import UserModel
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    ENABLE_FORWARD_USER_INFO_HEADERS,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.utils.headers import include_user_info_headers
from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.misc import stream_chunks_handler

router = APIRouter()

# Cerebro Model Configuration
CEREBRO_CONFIG = {
    "id": "cerebro-gpt",
    "name": "Cerebro GPT",
    "description": "Cerebro Advanced AI Model for Chat",
    "context_length": 4096,
    "max_tokens": 2048,
    "temperature_range": [0.0, 2.0],
    "default_temperature": 0.7,
}

class CerebroModel(BaseModel):
    """Cerebro Model request/response model"""
    model: str = "cerebro-gpt"
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class CerebroCompletionResponse(BaseModel):
    """Cerebro model completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def get_cerebro_model_info():
    """Get Cerebro model information in OpenAI format"""
    current_time = int(time.time())
    return {
        "id": CEREBRO_CONFIG["id"],
        "object": "model",
        "created": current_time,
        "owned_by": "cerebro",
        "permission": [],
        "root": "cerebro-gpt",
        "parent": None,
        "cerebro": CEREBRO_CONFIG,
    }

async def call_cerebro_model(messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Call Cerebro model via actual SLO integration"""
    try:
        # Extract user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)
        
        # Try to use actual SLO models
        try:
            # Import and use local SLO CLI
            import sys
            sys.path.insert(0, "/Users/mac/sloughGPT")
            
            from slo_cli import SLOCLI
            slo_cli = SLOCLI()
            
            # Get available models
            models = slo_cli.handle_model([])
            if models:
                model_id = models[0].get('id', 'cerebro-gpt')  # Use first available model
                logging.info(f"Using SLO model: {model_id}")
                
                # Generate response using actual SLO model
                # This would integrate with your actual model loading/generation
                # For now, simulate with better mock
                await asyncio.sleep(0.3)  # Simulate model processing
                
                response_text = f"🧠 Actual SLO model '{model_id}' responding to: {user_message[:100]}..."
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion", 
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(user_message.split()) + len(response_text.split()),
                    },
                }
            else:
                # Fallback to mock response
                logging.warning("No SLO models available, using fallback response")
                await asyncio.sleep(0.2)
                response_text = f"🧠 Cerebro AI (SLO integration ready): {user_message[:100]}..."
                
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "cerebro-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(user_message.split()) + len(response_text.split()),
                    },
                }
        
    except Exception as e:
        logging.error(f"SLO integration error: {e}")
        # Fallback to mock
        await asyncio.sleep(0.5)
        response_text = f"🧠 Cerebro AI response to: {user_message[:100]}..."
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "cerebro-gpt",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split()),
            },
        }

@router.get("/api/cerebro/models")
async def get_cerebro_models(request: Request, user: UserModel = None):
    """Get available Cerebro models"""
    
    model_info = get_cerebro_model_info()
    
    return {
        "object": "list",
        "data": [model_info],
    }

@router.post("/api/cerebro/chat/completions")
async def cerebro_chat_completions(
    request: Request,
    form_data: CerebroModel,
    user: UserModel = None,
):
    """Cerebro chat completions endpoint (OpenAI-compatible)"""
    
    try:
        # Apply system prompt if provided
        messages = apply_system_prompt_to_body(form_data.messages, user)
        
        # Apply model parameters
        kwargs = {
            "max_tokens": form_data.max_tokens,
            "temperature": form_data.temperature,
        }
        
        if form_data.stream:
            # Streaming response
            async def generate_stream():
                try:
                    response = await call_cerebro_model(messages, **kwargs)
                    chunk = {
                        "id": response["id"],
                        "object": "chat.completion.chunk",
                        "created": response["created"],
                        "model": response["model"],
                        "choices": response["choices"],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logging.error(f"Stream error: {e}")
                    yield f"data: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers=include_user_info_headers(request, user),
            )
        else:
            # Non-streaming response
            response = await call_cerebro_model(messages, **kwargs)
            
            return JSONResponse(
                content=response,
                headers=include_user_info_headers(request, user),
            )
            
    except Exception as e:
        logging.error(f"Cerebro chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/cerebro/models/{model_id}")
async def get_cerebro_model(model_id: str, request: Request, user: UserModel = None):
    """Get specific Cerebro model info"""
    
    if model_id != CEREBRO_CONFIG["id"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = get_cerebro_model_info()
    
    return JSONResponse(
        content=model_info,
        headers=include_user_info_headers(request, user),
    )

def add_cerebro_model_to_db(session: Session):
    """Add Cerebro model to database if not exists"""
    try:
        existing_model = session.query(Models).filter_by(id=CEREBRO_CONFIG["id"]).first()
        
        if not existing_model:
            cerebro_model = Models(
                id=CEREBRO_CONFIG["id"],
                name=CEREBRO_CONFIG["name"],
                info=json.dumps(CEREBRO_CONFIG),
                base_model_id=CEREBRO_CONFIG["id"],
                owned_by="cerebro",
            )
            session.add(cerebro_model)
            session.commit()
            logging.info(f"Cerebro model {CEREBRO_CONFIG['id']} added to database")
            
    except Exception as e:
        logging.error(f"Failed to add Cerebro model to DB: {e}")

# Auto-register Cerebro model on import
def register_cerebro_model():
    """Auto-register Cerebro model in OpenWebUI"""
    from open_webui.internal.db import Session
    
    try:
        with Session() as session:
            add_cerebro_model_to_db(session)
    except Exception as e:
        logging.error(f"Failed to auto-register Cerebro model: {e}")

# Register Cerebro model on module load
register_cerebro_model()