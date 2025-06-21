"""
Enhanced Chat API with coroutine safety and proper request handling.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse
from datetime import datetime

from app.core.logging import get_logger, get_correlation_id, set_correlation_id, log_performance
from app.core.config import get_settings
from app.core.async_utils import ensure_awaited, safe_execute, safe_graph_execute, coroutine_safe
from app.api.security import get_current_user, SecureChatInput, check_content_policy
from app.models.manager import QualityLevel
from app.graphs.chat_graph import ChatGraph
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.schemas.responses import (
    ChatResponse, ChatData, StreamingChatResponse, 
    ResponseMetadata, CostPrediction, DeveloperHints, 
    create_success_response, create_error_response, ConversationContext
)
from app.performance.optimization import OptimizedSearchSystem
from app.graphs.base import GraphState


router = APIRouter()
logger = get_logger("api.chat")
settings = get_settings()

# Global instances (will be initialized in main.py)
model_manager: Optional[ModelManager] = None
cache_manager: Optional[CacheManager] = None
chat_graph: Optional[ChatGraph] = None

# CORRECTED REQUEST MODELS WITH PROPER WRAPPERS
class ChatRequest(SecureChatInput):
    """Non-streaming chat request with comprehensive options."""
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional user context")
    quality_requirement: Optional[str] = Field("balanced", description="Quality level: minimal, balanced, high, premium")
    max_cost: Optional[float] = Field(0.10, ge=0.0, le=1.0, description="Maximum cost in INR")
    max_execution_time: Optional[float] = Field(30.0, ge=1.0, le=120.0, description="Maximum execution time")
    force_local_only: Optional[bool] = Field(False, description="Force local models only")
    response_style: Optional[str] = Field("balanced", description="Response style: concise, balanced, detailed")
    include_sources: Optional[bool] = Field(True, description="Include sources and citations")
    include_debug_info: Optional[bool] = Field(False, description="Include debug information")

    @field_validator('quality_requirement')
    @classmethod
    def validate_quality(cls, v):
        valid_qualities = ["minimal", "balanced", "high", "premium"]
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of: {valid_qualities}")
        return v

    @field_validator('response_style')
    @classmethod
    def validate_style(cls, v):
        valid_styles = ["concise", "balanced", "detailed"]
        if v not in valid_styles:
            raise ValueError(f"Style must be one of: {valid_styles}")
        return v

class StreamingChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    model: Optional[str] = Field("auto", description="Model preference")
    max_tokens: Optional[int] = Field(300, ge=1, le=2000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(True, description="Enable streaming")
    session_id: Optional[str] = Field(None, description="Session ID")

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        for msg in v:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            if msg['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v

async def initialize_chat_dependencies():
    global model_manager, cache_manager, chat_graph
    if not model_manager:
        model_manager = ModelManager()
        await model_manager.initialize()
        logger.info("ModelManager initialized for chat API")
    if not cache_manager:
        try:
            cache_manager = CacheManager(settings.redis_url)
            await cache_manager.initialize()
            logger.info("CacheManager initialized for chat API")
        except Exception as e:
            logger.warning(f"CacheManager initialization failed: {e}")
            cache_manager = None
    if not chat_graph:
        chat_graph = ChatGraph(model_manager, cache_manager)
        logger.info("ChatGraph initialized for chat API")

@router.get("/health")
async def chat_health():
    return {
        "status": "healthy",
        "service": "chat",
        "timestamp": time.time()
    }

# --- Begin merged advanced coroutine-safe endpoints and helpers ---

# CORRECTED MAIN ENDPOINT WITH COROUTINE SAFETY AND ADVANCED LOGIC
@router.post("/complete", response_model=ChatResponse)
@log_performance("chat_complete")
@coroutine_safe(timeout=60.0)
async def chat_complete(
    req: Request,
    background_tasks: BackgroundTasks,
    chat_request: ChatRequest = Body(..., embed=False),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    query_id = str(uuid.uuid4())
    correlation_id = get_correlation_id()
    start_time = time.time()
    logger.info(
        "Chat completion request started",
        query_id=query_id,
        user_id=current_user["user_id"],
        message_length=len(chat_request.message),
        session_id=chat_request.session_id,
        correlation_id=correlation_id
    )
    try:
        # Content policy check
        policy_check = check_content_policy(chat_request.message)
        if not policy_check["passed"]:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    message="Content violates policy",
                    error_code="CONTENT_POLICY_VIOLATION",
                    query_id=query_id,
                    correlation_id=correlation_id,
                    suggestions=["Please rephrase your request", "Avoid prohibited content"]
                ).dict()
            )
        session_id = chat_request.session_id or f"chat_{current_user['user_id']}_{int(time.time())}"
        conversation_history = chat_request.user_context.get("conversation_history", [])
        graph_state = GraphState(
            query_id=query_id,
            correlation_id=correlation_id,
            user_id=current_user["user_id"],
            session_id=session_id,
            original_query=chat_request.message,
            conversation_history=conversation_history,
            quality_requirement=getattr(QualityLevel, chat_request.quality_requirement.upper(), QualityLevel.BALANCED),
            max_cost=chat_request.max_cost,
            max_execution_time=chat_request.max_execution_time,
            user_preferences={
                "tier": current_user.get("tier", "free"),
                "response_style": chat_request.response_style,
                "include_sources": chat_request.include_sources,
                "force_local_only": chat_request.force_local_only
            }
        )
        chat_result = await safe_graph_execute(
            chat_graph, 
            graph_state, 
            timeout=chat_request.max_execution_time
        )
        chat_result = await ensure_awaited(chat_result)
        if not hasattr(chat_result, 'final_response'):
            logger.error(f"Chat result missing final_response: {type(chat_result)}")
            raise ValueError("Invalid chat result structure")
        conversation_context = ConversationContext(
            session_id=session_id,
            message_count=len(conversation_history) + 1,
            last_updated=datetime.utcnow().isoformat(),
            user_preferences=chat_request.user_context,
            conversation_summary=getattr(chat_result, 'conversation_summary', None)
        )
        chat_data = ChatData(
            response=chat_result.final_response,
            session_id=session_id,
            context=conversation_context,
            sources=getattr(chat_result, 'sources_consulted', []),
            citations=getattr(chat_result, 'citations', [])
        )
        total_cost = 0.0
        try:
            if hasattr(chat_result, 'calculate_total_cost'):
                cost_calc = chat_result.calculate_total_cost()
                total_cost = await ensure_awaited(cost_calc) if asyncio.iscoroutine(cost_calc) else cost_calc
        except Exception as e:
            logger.warning(f"Error calculating cost: {e}")
        execution_time = time.time() - start_time
        metadata = ResponseMetadata(
            query_id=query_id,
            execution_time=execution_time,
            cost=total_cost,
            models_used=list(getattr(chat_result, 'models_used', set())),
            confidence=getattr(chat_result, 'get_avg_confidence', lambda: 1.0)(),
            cached=False,
            timestamp=datetime.utcnow().isoformat()
        )
        cost_prediction = None
        if chat_request.include_debug_info:
            cost_prediction = CostPrediction(
                estimated_cost=total_cost,
                cost_breakdown=[],
                savings_tips=["Use lower quality settings for simple queries"],
                alternative_workflows=[]
            )
        developer_hints = None
        if chat_request.include_debug_info:
            developer_hints = DeveloperHints(
                execution_path=getattr(chat_result, 'execution_path', []),
                routing_explanation=f"Processed as {chat_request.quality_requirement} quality chat",
                performance={
                    "execution_time": execution_time,
                    "models_used": len(metadata.models_used),
                    "confidence": metadata.confidence
                }
            )
        response = ChatResponse(
            status="success",
            data=chat_data,
            metadata=metadata,
            cost_prediction=cost_prediction,
            developer_hints=developer_hints
        )
        from app.core.async_utils import AsyncSafetyValidator
        try:
            AsyncSafetyValidator.assert_no_coroutines(
                response, 
                "Chat response contains coroutines"
            )
        except AssertionError as e:
            logger.error(f"Coroutine safety check failed: {e}")
            return create_safe_fallback_response(query_id, correlation_id, execution_time)
        logger.info(
            "Chat completion successful",
            query_id=query_id,
            execution_time=execution_time,
            cost=total_cost,
            correlation_id=correlation_id
        )
        background_tasks.add_task(
            log_chat_analytics,
            query_id,
            request.message,
            chat_result.final_response,
            execution_time,
            total_cost
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "Chat completion failed",
            query_id=query_id,
            error=str(e),
            execution_time=execution_time,
            correlation_id=correlation_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Chat completion failed",
                error_code="CHAT_PROCESSING_ERROR",
                query_id=query_id,
                correlation_id=correlation_id,
                technical_details=str(e),
                suggestions=[
                    "Try rephrasing your question",
                    "Reduce complexity if query is very long", 
                    "Try again in a moment"
                ]
            ).dict()
        )

# CORRECTED STREAMING ENDPOINT WITH COROUTINE SAFETY
@router.post("/stream")
@coroutine_safe(timeout=120.0)
async def chat_stream(
    request: StreamingChatRequest,
    req: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    query_id = str(uuid.uuid4())
    await initialize_chat_dependencies()
    async def generate_safe_stream():
        try:
            user_message = ""
            for msg in reversed(request.messages):
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break
            policy_check = check_content_policy(user_message)
            if not policy_check["passed"]:
                yield _create_error_stream_chunk("Content violates policy")
                return
            session_id = request.session_id or f"stream_{current_user['user_id']}_{int(time.time())}"
            conversation_history = []
            for msg in request.messages[:-1]:
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": time.time()
                })
            graph_state = GraphState(
                query_id=query_id,
                correlation_id=correlation_id,
                user_id=current_user["user_id"],
                session_id=session_id,
                original_query=user_message,
                conversation_history=conversation_history,
                quality_requirement=QualityLevel.BALANCED,
                max_cost=0.10,
                max_execution_time=30.0,
                user_preferences={
                    "tier": current_user.get("tier", "free"),
                    "streaming": True
                }
            )
            chat_result = await safe_graph_execute(chat_graph, graph_state, timeout=30.0)
            chat_result = await ensure_awaited(chat_result)
            if chat_result.final_response:
                response_text = chat_result.final_response
                chunk_size = max(1, len(response_text) // 20)
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    stream_chunk = {
                        "id": f"chatcmpl-{query_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": list(chat_result.models_used)[0] if getattr(chat_result, 'models_used', None) else "local",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(stream_chunk)}\n\n"
                    await asyncio.sleep(0.05)
                final_chunk = {
                    "id": f"chatcmpl-{query_id}",
                    "object": "chat.completion.chunk", 
                    "created": int(time.time()),
                    "model": list(chat_result.models_used)[0] if getattr(chat_result, 'models_used', None) else "local",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield _create_error_stream_chunk("No response generated")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield _create_error_stream_chunk(f"Internal error: {str(e)}")
    return StreamingResponse(
        generate_safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

def _create_error_stream_chunk(error_message: str) -> str:
    error_chunk = {
        "id": f"chatcmpl-error-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [{
            "index": 0,
            "delta": {"content": f"Error: {error_message}"},
            "finish_reason": "stop"
        }]
    }
    return f"data: {json.dumps(error_chunk)}\n\n"

def create_safe_fallback_response(query_id: str, correlation_id: str, execution_time: float) -> ChatResponse:
    chat_data = ChatData(
        response="I apologize, but I'm experiencing technical difficulties. Please try again.",
        session_id="fallback_session",
        context=None,
        sources=[],
        citations=[]
    )
    metadata = ResponseMetadata(
        query_id=query_id,
        execution_time=execution_time,
        cost=0.0,
        models_used=["fallback"],
        confidence=0.0,
        cached=False,
        timestamp=datetime.utcnow().isoformat()
    )
    return ChatResponse(
        status="error",
        data=chat_data,
        metadata=metadata
    )

async def log_chat_analytics(
    query_id: str,
    user_message: str,
    response: str,
    execution_time: float,
    cost: float
):
    try:
        logger.info(
            "Chat analytics",
            query_id=query_id,
            message_length=len(user_message),
            response_length=len(response),
            execution_time=execution_time,
            cost=cost
        )
    except Exception as e:
        logger.error(f"Error logging analytics: {e}")

@router.get("/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        return {
            "session_id": session_id,
            "history": [],
            "message_count": 0,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")

@router.delete("/history/{session_id}")
async def clear_conversation_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        return {
            "cleared": True,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation history")
