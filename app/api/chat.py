# app/api/chat.py
"""
Chat API endpoints
Provides streaming and non-streaming chat interfaces
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
import json

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from app.graphs.base import GraphState
from app.graphs.chat_graph import create_chat_graph
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.api.middleware import get_current_user, check_rate_limit, track_cost
from app.schemas.requests import ChatRequest, ChatStreamRequest
from app.schemas.responses import ChatResponse, ErrorResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatCompletionChunk(BaseModel):
    """Single chunk in streaming response"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None


def get_model_manager(request: Request) -> ModelManager:
    """Get model manager from app state"""
    return request.app.state.model_manager


def get_cache_manager(request: Request) -> CacheManager:
    """Get cache manager from app state"""
    return request.app.state.cache_manager


@router.post("/stream", response_model=None)
async def chat_stream(
    request: ChatStreamRequest,
    req: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Streaming chat endpoint compatible with OpenAI API format
    Returns Server-Sent Events for real-time response streaming
    """
    try:
        # Rate limiting
        await check_rate_limit(current_user["id"], cache_manager)
        
        # Create graph state
        state = GraphState(
            query_id=str(uuid.uuid4()),
            user_id=current_user["id"],
            session_id=request.session_id,
            original_query=request.messages[-1]["content"],  # Last user message
            conversation_history=request.messages[:-1],  # All but last message
            cost_budget_remaining=await cache_manager.get_remaining_budget(current_user["id"]),
            max_execution_time=request.max_completion_time or 30.0,
            quality_requirement=request.quality_requirement or "balanced"
        )
        
        # Store user preferences
        if request.user_preferences:
            state.user_preferences.update(request.user_preferences)
        
        logger.info(
            "Starting streaming chat",
            query_id=state.query_id,
            user_id=current_user["id"],
            session_id=request.session_id
        )
        
        # Create and execute chat graph
        chat_graph = await create_chat_graph(model_manager, cache_manager)
        
        async def generate_response() -> AsyncGenerator[str, None]:
            """Generate streaming response"""
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(datetime.now().timestamp())
            
            try:
                # Execute graph (this could be made streaming in future)
                result_state = await chat_graph.execute(state)
                
                # Track cost
                total_cost = result_state.calculate_total_cost()
                await track_cost(current_user["id"], total_cost, cache_manager)
                
                if result_state.final_response:
                    # Split response into chunks for streaming effect
                    response_text = result_state.final_response
                    chunk_size = 50  # Characters per chunk
                    
                    for i in range(0, len(response_text), chunk_size):
                        chunk_text = response_text[i:i + chunk_size]
                        
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created_time,
                            model=result_state.response_metadata.get("model_used", "unknown"),
                            choices=[{
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None
                            }]
                        )
                        
                        yield f"data: {chunk.json()}\n\n"
                    
                    # Final chunk with metadata
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created_time,
                        model=result_state.response_metadata.get("model_used", "unknown"),
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        usage={
                            "total_tokens": len(response_text.split()),
                            "prompt_tokens": len(state.original_query.split()),
                            "completion_tokens": len(result_state.final_response.split()),
                            "total_cost": total_cost,
                            "execution_time": result_state.calculate_total_time(),
                            "confidence": result_state.get_avg_confidence()
                        }
                    )
                    
                    yield f"data: {final_chunk.json()}\n\n"
                    
                else:
                    # Error case
                    error_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created_time,
                        model="error",
                        choices=[{
                            "index": 0,
                            "delta": {"content": "I apologize, but I encountered an error processing your request."},
                            "finish_reason": "error"
                        }]
                    )
                    
                    yield f"data: {error_chunk.json()}\n\n"
                
                # End of stream
                yield "data: [DONE]\n\n"
                
                logger.info(
                    "Streaming chat completed",
                    query_id=state.query_id,
                    total_cost=total_cost,
                    execution_time=result_state.calculate_total_time()
                )
                
            except Exception as e:
                logger.error(
                    "Streaming chat error",
                    query_id=state.query_id,
                    error=str(e),
                    exc_info=e
                )
                
                # Send error chunk
                error_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model="error",
                    choices=[{
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "error"
                    }]
                )
                
                yield f"data: {error_chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Query-ID": state.query_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat stream endpoint error: {e}", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/complete", response_model=ChatResponse)
async def chat_complete(
    request: ChatRequest,
    req: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Non-streaming chat completion endpoint
    Returns complete response at once
    """
    try:
        # Rate limiting
        await check_rate_limit(current_user["id"], cache_manager)
        
        # Create graph state
        state = GraphState(
            query_id=str(uuid.uuid4()),
            user_id=current_user["id"], 
            session_id=request.session_id,
            original_query=request.message,
            conversation_history=request.conversation_history or [],
            cost_budget_remaining=await cache_manager.get_remaining_budget(current_user["id"]),
            max_execution_time=request.constraints.max_time if request.constraints else 30.0,
            quality_requirement=request.constraints.quality_requirement if request.constraints else "balanced"
        )
        
        # Store user preferences
        if request.context and request.context.user_preferences:
            state.user_preferences.update(request.context.user_preferences)
        
        logger.info(
            "Starting chat completion",
            query_id=state.query_id,
            user_id=current_user["id"],
            message_length=len(request.message)
        )
        
        # Create and execute chat graph
        chat_graph = await create_chat_graph(model_manager, cache_manager)
        result_state = await chat_graph.execute(state)
        
        # Track cost
        total_cost = result_state.calculate_total_cost()
        await track_cost(current_user["id"], total_cost, cache_manager)
        
        # Build response
        response = ChatResponse(
            status="success" if result_state.final_response else "error",
            data={
                "response": result_state.final_response or "I apologize, but I couldn't generate a response.",
                "session_id": request.session_id
            },
            metadata={
                "query_id": state.query_id,
                "execution_time": result_state.calculate_total_time(),
                "cost": total_cost,
                "models_used": list(set(
                    step for step in result_state.execution_path 
                    if step in ["phi:mini", "llama2:7b", "mistral:7b", "llama2:13b", "codellama"]
                )),
                "confidence": result_state.get_avg_confidence(),
                "cached": len(result_state.cache_hits) > 0
            },
            cost_prediction={
                "estimated_cost": total_cost,
                "cost_breakdown": [
                    {"step": step, "cost": cost}
                    for step, cost in result_state.costs_incurred.items()
                ],
                "savings_tips": _generate_savings_tips(result_state)
            },
            developer_hints={
                "execution_path": result_state.execution_path,
                "intent_detected": result_state.intermediate_results.get("intent"),
                "routing_explanation": f"Routed through {' -> '.join(result_state.execution_path)}",
                "performance": {
                    "cache_hits": len(result_state.cache_hits),
                    "total_steps": len(result_state.execution_path),
                    "avg_confidence": result_state.get_avg_confidence()
                }
            }
        )
        
        logger.info(
            "Chat completion finished",
            query_id=state.query_id,
            total_cost=total_cost,
            execution_time=result_state.calculate_total_time(),
            success=bool(result_state.final_response)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat complete endpoint error: {e}", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    req: Request,
    cache_manager: CacheManager = Depends(get_cache_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get conversation history for a session"""
    try:
        history = await cache_manager.get_conversation_history(session_id)
        
        return {
            "status": "success",
            "data": {
                "session_id": session_id,
                "history": history or [],
                "length": len(history) if history else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@router.delete("/history/{session_id}")
async def clear_conversation_history(
    session_id: str,
    req: Request,
    cache_manager: CacheManager = Depends(get_cache_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Clear conversation history for a session"""
    try:
        # Clear from cache
        conv_key = f"conv:{session_id}"
        await cache_manager.batch_delete([conv_key])
        
        return {
            "status": "success",
            "data": {
                "session_id": session_id,
                "cleared": True
            }
        }
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation history")


@router.get("/sessions")
async def list_user_sessions(
    req: Request,
    cache_manager: CacheManager = Depends(get_cache_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List user's active chat sessions"""
    try:
        # This is a simplified implementation
        # In production, you'd want to maintain a proper session registry
        
        # For now, return empty sessions list
        # This could be enhanced to scan cache for user's conversation keys
        
        return {
            "status": "success", 
            "data": {
                "sessions": [],
                "total_sessions": 0
            }
        }
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


def _generate_savings_tips(state: GraphState) -> list[str]:
    """Generate cost-saving tips based on execution"""
    tips = []
    
    # Check if premium models were used unnecessarily
    if state.quality_requirement != "premium" and "llama2:13b" in state.execution_path:
        tips.append("Consider using 'balanced' quality for similar queries to reduce processing time")
    
    # Check for cache opportunities
    if len(state.cache_hits) == 0:
        tips.append("Similar queries might be cached for faster responses in the future")
    
    # Check for simple queries using complex models
    intent = state.intermediate_results.get("intent")
    if intent == "simple_chat" and any(model in state.execution_path for model in ["mistral:7b", "llama2:13b"]):
        tips.append("Simple questions can often be answered with faster models")
    
    return tips


# Error handlers specific to chat endpoints
@router.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return ErrorResponse(
        status="error",
        message=f"Invalid input: {str(exc)}",
        error_code="INVALID_INPUT"
    )


@router.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    return ErrorResponse(
        status="error", 
        message="Request timed out. Please try again with a simpler query.",
        error_code="TIMEOUT"
    )
