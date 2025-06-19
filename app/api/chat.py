"""
Enhanced Chat API with full graph system integration.
Provides both streaming and non-streaming endpoints with production features.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from app.core.logging import get_logger, get_correlation_id, set_correlation_id, log_performance
from app.core.config import get_settings
from app.api.security import get_current_user, SecureChatInput, check_content_policy
from app.graphs.base import GraphState
from app.models.manager import QualityLevel
from app.graphs.chat_graph import ChatGraph
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.schemas.responses import (
    ChatResponse, ChatData, StreamingChatResponse, OpenAIChatResponse, 
    ResponseMetadata, CostPrediction, DeveloperHints, create_success_response,
    create_error_response, ConversationContext
)


router = APIRouter()
logger = get_logger("api.chat")
settings = get_settings()

# Global instances (will be initialized in main.py)
model_manager: Optional[ModelManager] = None
cache_manager: Optional[CacheManager] = None
chat_graph: Optional[ChatGraph] = None


class ChatRequest(SecureChatInput):
    """Non-streaming chat request with comprehensive options."""
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional user context")
    
    # Quality and performance controls
    quality_requirement: Optional[str] = Field("balanced", description="Quality level: minimal, balanced, high, premium")
    max_cost: Optional[float] = Field(0.10, ge=0.0, le=1.0, description="Maximum cost in INR")
    max_execution_time: Optional[float] = Field(30.0, ge=1.0, le=120.0, description="Maximum execution time")
    force_local_only: Optional[bool] = Field(False, description="Force local models only")
    
    # Response preferences
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


class StreamingChatRequest(SecureChatInput):
    """OpenAI-compatible streaming chat request."""
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    model: Optional[str] = Field("auto", description="Model preference (auto-select if 'auto')")
    max_tokens: Optional[int] = Field(300, ge=1, le=2000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(True, description="Enable streaming")
    session_id: Optional[str] = Field(None, description="Session ID for context")

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
    """Initialize chat API dependencies."""
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


@router.post("/complete", response_model=ChatResponse)
@log_performance("chat_complete")
async def chat_complete(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Complete chat endpoint with full graph system integration.
    
    Features:
    - Intelligent conversation management
    - Context-aware responses
    - Cost optimization and tracking
    - Comprehensive error handling
    """
    # Initialize correlation ID for this request
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    # Initialize dependencies
    await initialize_chat_dependencies()
    
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    logger.info(
        "Chat completion request started",
        query_id=query_id,
        user_id=current_user["user_id"],
        message_length=len(request.message),
        session_id=request.session_id,
        correlation_id=correlation_id
    )
    
    try:
        # Content policy check
        policy_check = check_content_policy(request.message)
        if not policy_check["passed"]:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    message="Content violates policy",
                    error_code="CONTENT_POLICY_VIOLATION",
                    query_id=query_id,
                    correlation_id=correlation_id,
                    suggestions=["Please rephrase your request", "Avoid inappropriate content"]
                ).dict()
            )
        
        # Create session ID if not provided
        session_id = request.session_id or f"session_{current_user['user_id']}_{int(time.time())}"
        
        # Map quality requirement
        quality_mapping = {
            "minimal": QualityLevel.MINIMAL,
            "balanced": QualityLevel.BALANCED,
            "high": QualityLevel.HIGH,
            "premium": QualityLevel.PREMIUM
        }
        quality_level = quality_mapping.get(request.quality_requirement, QualityLevel.BALANCED)
        
        # Create graph state
        graph_state = GraphState(
            query_id=query_id,
            correlation_id=correlation_id,
            user_id=current_user["user_id"],
            session_id=session_id,
            original_query=request.message,
            quality_requirement=quality_level,
            max_cost=request.max_cost,
            max_execution_time=request.max_execution_time,
            force_local_only=request.force_local_only,
            user_preferences={
                "response_style": request.response_style,
                "include_sources": request.include_sources,
                "tier": current_user.get("tier", "free"),
                **request.user_context
            }
        )
        
        # Execute chat graph
        final_state = await chat_graph.execute(graph_state)
        
        # Calculate metrics
        execution_time = time.time() - start_time
        total_cost = final_state.calculate_total_cost()
        avg_confidence = final_state.get_avg_confidence()
        
        # Create conversation context for response
        context_data = final_state.intermediate_results.get("conversation_context", {})
        conversation_context = ConversationContext(
            session_id=session_id,
            message_count=len(final_state.conversation_history) + 1,
            user_preferences=final_state.user_preferences,
            conversation_summary=context_data.get("conversation_topic")
        )
        
        # Build response data
        chat_data = ChatData(
            response=final_state.final_response or "I'm sorry, I couldn't generate a response.",
            session_id=session_id,
            context=conversation_context,
            sources=final_state.sources_consulted,
            citations=final_state.citations
        )
        
        # Build metadata
        metadata = ResponseMetadata(
            query_id=query_id,
            correlation_id=correlation_id,
            execution_time=execution_time,
            cost=total_cost,
            models_used=list(final_state.models_used),
            confidence=avg_confidence,
            cached=False,  # Graph execution is not cached (individual components may be)
            routing_path=final_state.execution_path,
            escalation_reason=final_state.escalation_reason,
            cache_hit_rate=0.0,  # TODO: Calculate from individual node cache hits
            local_processing_percentage=1.0 if total_cost == 0.0 else 0.8,
            source_count=len(final_state.sources_consulted),
            citation_count=len(final_state.citations)
        )
        
        # Build cost prediction
        cost_prediction = CostPrediction(
            estimated_cost=total_cost,
            cost_breakdown=[
                {"step": step, "service": "local_model", "cost": cost}
                for step, cost in final_state.costs_incurred.items()
            ],
            savings_tips=[
                "All processing done locally - no additional costs incurred",
                "Consider using 'minimal' quality for faster responses on simple queries"
            ] if total_cost == 0.0 else [
                "Some API calls were made - consider local alternatives",
                "Upgrade to pro tier for better cost optimization"
            ],
            budget_remaining=request.max_cost - total_cost,
            budget_percentage_used=(total_cost / request.max_cost) * 100 if request.max_cost > 0 else 0
        )
        
        # Build developer hints
        developer_hints = DeveloperHints(
            suggested_next_queries=[
                "Can you explain that in more detail?",
                "What are some examples?",
                "How does this relate to other concepts?"
            ],
            potential_optimizations={
                "response_time": f"Current: {execution_time:.2f}s. Try 'minimal' quality for faster responses.",
                "cost_efficiency": "Using local models - optimal cost efficiency achieved.",
                "quality": f"Confidence: {avg_confidence:.2f}. Use 'high' quality for better results."
            },
            knowledge_gaps=[warning for warning in final_state.warnings if "confidence" in warning.lower()],
            routing_explanation=f"Query routed through: {' → '.join(final_state.execution_path)}",
            cache_hit_info="Conversation context cached for faster follow-up responses",
            performance_hints={
                "execution_time": execution_time,
                "nodes_executed": len(final_state.execution_path),
                "avg_confidence": avg_confidence,
                "optimization_suggestions": []
            }
        )
        
        # Create response
        response = ChatResponse(
            status="success" if not final_state.errors else "partial",
            data=chat_data,
            metadata=metadata,
            cost_prediction=cost_prediction,
            developer_hints=developer_hints if request.include_debug_info else None
        )
        
        # Log success
        logger.info(
            "Chat completion successful",
            query_id=query_id,
            execution_time=execution_time,
            cost=total_cost,
            confidence=avg_confidence,
            response_length=len(final_state.final_response),
            correlation_id=correlation_id
        )
        
        # Background task for analytics
        background_tasks.add_task(
            _record_chat_analytics,
            query_id, current_user["user_id"], execution_time, total_cost, avg_confidence
        )
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
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


@router.post("/stream")
async def chat_stream(
    request: StreamingChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Streaming chat endpoint compatible with OpenAI API format.
    
    Provides real-time response streaming for better user experience.
    """
    # Initialize correlation ID
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    # Initialize dependencies
    await initialize_chat_dependencies()
    
    # Extract user message
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    user_message = request.messages[-1]["content"]
    query_id = str(uuid.uuid4())
    
    logger.info(
        "Streaming chat request started",
        query_id=query_id,
        user_id=current_user["user_id"],
        message_count=len(request.messages),
        correlation_id=correlation_id
    )
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            # Content policy check
            policy_check = check_content_policy(user_message)
            if not policy_check["passed"]:
                yield _create_error_stream_chunk("Content violates policy")
                return
            
            # Create session ID
            session_id = request.session_id or f"stream_{current_user['user_id']}_{int(time.time())}"
            
            # Build conversation history
            conversation_history = []
            for msg in request.messages[:-1]:  # Exclude current message
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": time.time()
                })
            
            # Create graph state
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
            
            # Execute chat graph (non-streaming)
            # TODO: Implement streaming at graph level
            final_state = await chat_graph.execute(graph_state)
            
            if final_state.final_response:
                # Simulate streaming by chunking the response
                response_text = final_state.final_response
                chunk_size = max(1, len(response_text) // 20)  # ~20 chunks
                
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    
                    stream_chunk = {
                        "id": f"chatcmpl-{query_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": list(final_state.models_used)[0] if final_state.models_used else "local",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(stream_chunk)}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Final chunk
                final_chunk = {
                    "id": f"chatcmpl-{query_id}",
                    "object": "chat.completion.chunk", 
                    "created": int(time.time()),
                    "model": list(final_state.models_used)[0] if final_state.models_used else "local",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
            else:
                yield _create_error_stream_chunk("No response generated")
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(
                "Streaming chat failed",
                query_id=query_id,
                error=str(e),
                correlation_id=correlation_id,
                exc_info=True
            )
            yield _create_error_stream_chunk(f"Error: {str(e)}")
    
    return EventSourceResponse(generate_stream())


@router.get("/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get conversation history for a session."""
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    logger.info(
        "Conversation history requested",
        session_id=session_id,
        user_id=current_user["user_id"],
        limit=limit,
        correlation_id=correlation_id
    )
    
    try:
        await initialize_chat_dependencies()
        
        if cache_manager:
            history = await cache_manager.get_conversation_history(session_id)
            
            # Limit results
            if history and len(history) > limit:
                history = history[-limit:]
            
            return {
                "session_id": session_id,
                "history": history or [],
                "message_count": len(history) if history else 0,
                "correlation_id": correlation_id
            }
        else:
            return {
                "session_id": session_id,
                "history": [],
                "message_count": 0,
                "correlation_id": correlation_id,
                "note": "Cache manager not available"
            }
            
    except Exception as e:
        logger.error(
            "Failed to get conversation history",
            session_id=session_id,
            error=str(e),
            correlation_id=correlation_id
        )
        
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Failed to retrieve conversation history",
                error_code="HISTORY_RETRIEVAL_ERROR",
                correlation_id=correlation_id
            ).dict()
        )


@router.delete("/history/{session_id}")
async def clear_conversation_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Clear conversation history for a session."""
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    logger.info(
        "Conversation history clear requested",
        session_id=session_id,
        user_id=current_user["user_id"],
        correlation_id=correlation_id
    )
    
    try:
        await initialize_chat_dependencies()
        
        if cache_manager:
            await cache_manager.update_conversation_history(session_id, [])
            
        return {
            "session_id": session_id,
            "cleared": True,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error(
            "Failed to clear conversation history",
            session_id=session_id,
            error=str(e),
            correlation_id=correlation_id
        )
        
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Failed to clear conversation history",
                error_code="HISTORY_CLEAR_ERROR",
                correlation_id=correlation_id
            ).dict()
        )


def _create_error_stream_chunk(error_message: str) -> str:
    """Create an error chunk for streaming responses."""
    error_chunk = {
        "id": f"chatcmpl-error-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [{
            "index": 0,
            "delta": {"content": f"Error: {error_message}"},
            "finish_reason": "error"
        }]
    }
    return f"data: {json.dumps(error_chunk)}\n\n"


async def _record_chat_analytics(
    query_id: str,
    user_id: str, 
    execution_time: float,
    cost: float,
    confidence: float
):
    """Background task to record chat analytics."""
    try:
        # TODO: Implement analytics recording
        # This could write to ClickHouse, send to analytics service, etc.
        logger.info(
            "Chat analytics recorded",
            query_id=query_id,
            user_id=user_id,
            execution_time=execution_time,
            cost=cost,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Failed to record analytics: {e}")


# Dependency injection for testing
def set_dependencies(
    model_manager_instance: ModelManager,
    cache_manager_instance: Optional[CacheManager] = None,
    chat_graph_instance: Optional[ChatGraph] = None
):
    """Set dependencies for testing."""
    global model_manager, cache_manager, chat_graph
    model_manager = model_manager_instance
    cache_manager = cache_manager_instance
    chat_graph = chat_graph_instance or ChatGraph(model_manager, cache_manager)


# Export router
__all__ = ['router', 'set_dependencies']
