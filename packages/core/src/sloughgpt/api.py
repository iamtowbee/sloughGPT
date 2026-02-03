"""Main API server for SloughGPT Enterprise Framework."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from .user_management import get_user_manager, UserRole
from .cost_optimization import get_cost_optimizer, CostMetricType
from .data_learning import DatasetPipeline
from .reasoning_engine import ReasoningEngine
from .auth import AuthService
from .monitoring import MonitoringService


# Initialize components
user_manager = get_user_manager()
cost_optimizer = get_cost_optimizer()
auth_service = AuthService(user_manager)
monitoring_service = MonitoringService()

# FastAPI app
app = FastAPI(
    title="SloughGPT Enterprise API",
    description="Enterprise AI Framework API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    user_id: Optional[int] = None

class GenerateResponse(BaseModel):
    response: str
    tokens_used: int
    cost: float
    reasoning_steps: Optional[List[Dict[str, Any]]] = None

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: UserResponse

class DatasetAddRequest(BaseModel):
    name: str
    path: str
    format: str
    type: str = "file"

class ReasonRequest(BaseModel):
    prompt: str
    context_id: Optional[str] = None

class ReasonResponse(BaseModel):
    final_answer: str
    confidence: float
    reasoning_steps: List[Dict[str, Any]]
    context_id: str


# Middleware for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_service.validate_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return payload


# Middleware for rate limiting
async def check_rate_limit(user_payload: Dict[str, Any] = Depends(get_current_user)):
    if auth_service.is_rate_limited(user_payload["user_id"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return user_payload


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "user_management": "active",
            "cost_optimization": "active",
            "data_learning": "active",
            "reasoning_engine": "active",
            "monitoring": "active"
        }
    }


# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreateRequest):
    """Register a new user."""
    try:
        user_role = UserRole(user_data.role.lower())
        result = user_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_role
        )
        
        return UserResponse(
            id=result["user"]["id"],
            username=result["user"]["username"],
            email=result["user"]["email"],
            role=result["user"]["role"]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """Authenticate user and return tokens."""
    result = auth_service.authenticate(login_data.username, login_data.password)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    return LoginResponse(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        token_type=result["token_type"],
        user=UserResponse(
            id=result["user"]["id"],
            username=result["user"]["username"],
            email=result["user"]["email"],
            role=result["user"]["role"]
        )
    )


@app.post("/auth/logout")
async def logout(user_payload: Dict[str, Any] = Depends(get_current_user)):
    """Logout user."""
    # Token would be extracted from header and blacklisted
    return {"message": "Logged out successfully"}


# Text generation endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    user_payload: Dict[str, Any] = Depends(check_rate_limit)
):
    """Generate text using SloughGPT."""
    try:
        user_id = request.user_id or user_payload["user_id"]
        
        # Track cost before generation
        cost_optimizer.track_metric(
            user_id=user_id,
            metric_type=CostMetricType.TOKEN_INFERENCE,
            amount=request.max_tokens,
            model_name="sloughgpt-base"
        )
        
        # Use reasoning engine for generation
        reasoning_engine = ReasoningEngine()
        context_id = reasoning_engine.create_context(
            user_id=user_id,
            prompt=request.prompt,
            metadata={"max_tokens": request.max_tokens, "temperature": request.temperature}
        )
        
        result = reasoning_engine.reason(context_id)
        
        # Track metrics
        monitoring_service.increment_counter(
            "api_requests_total",
            labels={"endpoint": "/generate", "user_role": user_payload["role"]}
        )
        
        monitoring_service.record_histogram(
            "generation_tokens",
            result.final_answer.count(" "),
            labels={"model": "sloughgpt-base"}
        )
        
        # Calculate cost
        tokens_used = len(result.final_answer.split())
        cost = cost_optimizer.cost_per_token.get("sloughgpt-base", 0.000001) * tokens_used
        
        return GenerateResponse(
            response=result.final_answer,
            tokens_used=tokens_used,
            cost=cost,
            reasoning_steps=[
                {
                    "step": step.description,
                    "confidence": step.confidence,
                    "corrections": step.corrections
                }
                for step in result.reasoning_steps
            ]
        )
    
    except Exception as e:
        monitoring_service.increment_counter(
            "api_errors_total",
            labels={"endpoint": "/generate", "error_type": type(e).__name__}
        )
        raise HTTPException(status_code=500, detail=str(e))


# Reasoning endpoints
@app.post("/reason", response_model=ReasonResponse)
async def reason_about_prompt(
    request: ReasonRequest,
    user_payload: Dict[str, Any] = Depends(check_rate_limit)
):
    """Perform advanced reasoning on a prompt."""
    try:
        reasoning_engine = ReasoningEngine()
        
        if request.context_id and reasoning_engine.active_contexts.get(request.context_id):
            context_id = request.context_id
        else:
            context_id = reasoning_engine.create_context(
                user_id=user_payload["user_id"],
                prompt=request.prompt
            )
        
        result = reasoning_engine.reason(context_id)
        
        return ReasonResponse(
            final_answer=result.final_answer,
            confidence=result.confidence,
            reasoning_steps=[
                {
                    "step": step.description,
                    "confidence": step.confidence,
                    "corrections": step.corrections or []
                }
                for step in result.reasoning_steps
            ],
            context_id=context_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dataset management endpoints
@app.post("/datasets/add")
async def add_dataset(
    request: DatasetAddRequest,
    user_payload: Dict[str, Any] = Depends(check_rate_limit)
):
    """Add a new dataset to the learning pipeline."""
    try:
        pipeline = DatasetPipeline()
        
        source_id = pipeline.add_source(
            name=request.name,
            path=request.path,
            format=request.format,
            type=request.type,
            metadata={"added_by": user_payload["user_id"]}
        )
        
        return {
            "source_id": source_id,
            "message": "Dataset source added successfully",
            "status": "pending_processing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/search")
async def search_datasets(
    query: str,
    k: int = 5,
    user_payload: Dict[str, Any] = Depends(get_current_user)
):
    """Search across all datasets."""
    try:
        pipeline = DatasetPipeline()
        results = pipeline.search_knowledge(query, k)
        
        return {
            "query": query,
            "results": [
                {
                    "id": result.id,
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "total_found": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/learn")
async def start_learning(
    user_payload: Dict[str, Any] = Depends(check_rate_limit)
):
    """Start autonomous learning from all data sources."""
    try:
        pipeline = DatasetPipeline()
        
        job_id = await pipeline.start_learning({
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "quality_threshold": 0.85,
            "similarity_threshold": 0.8
        })
        
        return {
            "job_id": job_id,
            "message": "Learning pipeline started",
            "status": "running"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Cost management endpoints
@app.get("/costs/analysis")
async def get_cost_analysis(
    days: int = 30,
    user_payload: Dict[str, Any] = Depends(get_current_user)
):
    """Get cost analysis for the user."""
    try:
        analysis = cost_optimizer.analyze_usage_patterns(
            user_payload["user_id"], 
            days=days
        )
        
        return {
            "period_days": days,
            "total_cost": analysis.total_cost,
            "avg_daily_cost": analysis.avg_daily_cost,
            "metrics_by_type": analysis.metrics_by_type,
            "recommendations": analysis.recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/costs/budget")
async def set_budget(
    monthly: Optional[float] = None,
    daily: Optional[float] = None,
    hourly: Optional[float] = None,
    user_payload: Dict[str, Any] = Depends(get_current_user)
):
    """Set budget limits for the user."""
    try:
        cost_optimizer.set_user_budget(
            user_payload["user_id"],
            monthly=monthly,
            daily=daily,
            hourly=hourly
        )
        
        return {
            "message": "Budget limits updated",
            "monthly": monthly,
            "daily": daily,
            "hourly": hourly
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@app.get("/monitoring/metrics")
async def get_metrics(user_payload: Dict[str, Any] = Depends(get_current_user)):
    """Get system metrics (admin only)."""
    if user_payload["role"] != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        dashboard_data = monitoring_service.get_dashboard_data()
        return dashboard_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/export/{format}")
async def export_metrics(
    format: str,
    user_payload: Dict[str, Any] = Depends(get_current_user)
):
    """Export metrics in specified format (admin only)."""
    if user_payload["role"] != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        if format not in ["prometheus", "json"]:
            raise HTTPException(
                status_code=400,
                detail="Format must be 'prometheus' or 'json'"
            )
        
        metrics_data = monitoring_service.export_metrics(format)
        
        if format == "json":
            return {"metrics": metrics_data}
        else:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=metrics_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logging.info("SloughGPT API Server starting up...")
    
    # Set up monitoring metrics
    monitoring_service.create_alert_rule(
        "high_error_rate",
        "api_errors_total",
        "gt",
        10,
        60,
        "warning"
    )
    
    monitoring_service.create_alert_rule(
        "high_response_time",
        "api_response_time",
        "gt",
        5.0,
        60,
        "warning"
    )
    
    logging.info("SloughGPT API Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logging.info("SloughGPT API Server shutting down...")
    monitoring_service.cleanup()
    logging.info("SloughGPT API Server shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )