"""
Real response schemas with comprehensive metadata, cost tracking, and developer hints.
Replaces dummy responses.py
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class CostBreakdown(BaseModel):
    """Individual cost component breakdown."""
    step: str = Field(..., description="Processing step name")
    service: str = Field(..., description="Service or model used")
    cost: float = Field(..., description="Cost in INR")
    duration_ms: Optional[float] = Field(None, description="Step duration in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed if applicable")


class CostPrediction(BaseModel):
    """Cost prediction and optimization recommendations."""
    estimated_cost: float = Field(..., description="Total estimated cost in INR")
    cost_breakdown: List[CostBreakdown] = Field(default_factory=list, description="Detailed cost breakdown")
    savings_tips: List[str] = Field(default_factory=list, description="Cost optimization suggestions")
    alternative_workflows: Optional[List[Dict[str, Any]]] = Field(
        None, description="Alternative cheaper workflows"
    )
    budget_remaining: Optional[float] = Field(None, description="Remaining user budget in INR")
    budget_percentage_used: Optional[float] = Field(None, description="Percentage of monthly budget used")


class DeveloperHints(BaseModel):
    """Developer experience enhancements and debugging information."""
    suggested_next_queries: List[str] = Field(default_factory=list, description="Suggested follow-up queries")
    potential_optimizations: Dict[str, str] = Field(default_factory=dict, description="Performance optimization suggestions")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Areas where confidence was low")
    routing_explanation: Optional[str] = Field(None, description="Why this routing path was chosen")
    cache_hit_info: Optional[str] = Field(None, description="Cache performance information")
    performance_hints: Optional[Dict[str, Any]] = Field(None, description="Performance-related hints")


class ResponseMetadata(BaseModel):
    """Comprehensive response metadata for debugging and analytics."""
    query_id: str = Field(..., description="Unique query identifier")
    correlation_id: str = Field(..., description="Request correlation ID for tracing")
    execution_time: float = Field(..., description="Total execution time in seconds")
    cost: float = Field(..., description="Total cost in INR")
    models_used: List[str] = Field(default_factory=list, description="Models used in processing")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    cached: bool = Field(False, description="Whether response was served from cache")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")
    
    # Execution path tracking
    routing_path: List[str] = Field(default_factory=list, description="Graph nodes executed")
    escalation_reason: Optional[str] = Field(None, description="Why request was escalated to premium models")
    
    # Performance metrics
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate during execution")
    local_processing_percentage: Optional[float] = Field(None, description="Percentage processed locally vs API")
    
    # Quality metrics
    source_count: Optional[int] = Field(None, description="Number of sources consulted")
    citation_count: Optional[int] = Field(None, description="Number of citations included")


class BaseResponse(BaseModel):
    """Base response model with standard metadata."""
    status: str = Field(..., description="Response status: success, error, partial")
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    cost_prediction: Optional[CostPrediction] = Field(None, description="Cost analysis and predictions")
    developer_hints: Optional[DeveloperHints] = Field(None, description="Developer experience enhancements")


class ChatMessage(BaseModel):
    """Individual chat message in conversation."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")


class ConversationContext(BaseModel):
    """Conversation context and history."""
    session_id: str = Field(..., description="Conversation session ID")
    message_count: int = Field(0, description="Number of messages in conversation")
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="Inferred user preferences")
    conversation_summary: Optional[str] = Field(None, description="AI-generated conversation summary")


class ChatData(BaseModel):
    """Chat response data structure."""
    response: str = Field(..., description="Generated response text")
    session_id: str = Field(..., description="Conversation session ID")
    context: Optional[ConversationContext] = Field(None, description="Conversation context")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used in response")
    citations: List[str] = Field(default_factory=list, description="Citation references")


class ChatResponse(BaseResponse):
    """Complete chat completion response."""
    data: ChatData = Field(..., description="Chat response data")


class StreamingChatResponse(BaseModel):
    """OpenAI-compatible streaming chat response."""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion.chunk", description="Response object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used for generation")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")


class SearchData(BaseModel):
    """Search response data structure."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    summary: Optional[str] = Field(None, description="AI-generated summary of results")
    total_results: int = Field(0, description="Total number of results found")
    search_time: float = Field(..., description="Search execution time")
    sources_consulted: List[str] = Field(default_factory=list, description="Search providers used")


class SearchResponse(BaseResponse):
    """Complete search response."""
    data: SearchData = Field(..., description="Search response data")


class HealthStatus(BaseModel):
    """Service health status."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(default_factory=dict, description="Individual component health")
    version: str = Field(..., description="Application version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    last_check: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ErrorDetails(BaseModel):
    """Detailed error information."""
    error_code: str = Field(..., description="Machine-readable error code")
    error_type: str = Field(..., description="Error classification")
    user_message: str = Field(..., description="User-friendly error message")
    technical_details: Optional[str] = Field(None, description="Technical error details")
    suggestions: List[str] = Field(default_factory=list, description="Suggested solutions")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    error_details: Optional[ErrorDetails] = Field(None, description="Detailed error information")
    query_id: Optional[str] = Field(None, description="Query ID for tracking")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsData(BaseModel):
    """System metrics data."""
    cache: Dict[str, Any] = Field(default_factory=dict, description="Cache performance metrics")
    models: Dict[str, Any] = Field(default_factory=dict, description="Model performance metrics")
    system: Dict[str, Any] = Field(default_factory=dict, description="System performance metrics")
    costs: Dict[str, Any] = Field(default_factory=dict, description="Cost metrics")


class MetricsResponse(BaseResponse):
    """System metrics response."""
    metrics: MetricsData = Field(..., description="System metrics")


class UserStats(BaseModel):
    """User usage statistics."""
    user_id: str = Field(..., description="User identifier")
    requests_today: int = Field(0, description="Requests made today")
    requests_this_month: int = Field(0, description="Requests made this month")
    cost_today: float = Field(0.0, description="Cost incurred today")
    cost_this_month: float = Field(0.0, description="Cost incurred this month")
    budget_remaining: float = Field(0.0, description="Remaining monthly budget")
    tier: str = Field("free", description="User tier")
    rate_limit_remaining: int = Field(0, description="Remaining rate limit")


class UserStatsResponse(BaseResponse):
    """User statistics response."""
    stats: UserStats = Field(..., description="User statistics")


# OpenAI-compatible response models for streaming
class OpenAIChoice(BaseModel):
    """OpenAI-compatible choice object."""
    index: int = Field(..., description="Choice index")
    delta: Optional[Dict[str, Any]] = Field(None, description="Delta content for streaming")
    message: Optional[Dict[str, Any]] = Field(None, description="Complete message for non-streaming")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Completion ID")
    object: str = Field(..., description="Response object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[OpenAIChoice] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")


# Utility functions for response creation
def create_success_response(
    data: Any,
    query_id: str,
    correlation_id: str,
    execution_time: float,
    cost: float,
    models_used: List[str],
    confidence: float,
    cached: bool = False
) -> Dict[str, Any]:
    """Create a standardized success response."""
    return {
        "status": "success",
        "data": data,
        "metadata": ResponseMetadata(
            query_id=query_id,
            correlation_id=correlation_id,
            execution_time=execution_time,
            cost=cost,
            models_used=models_used,
            confidence=confidence,
            cached=cached
        )
    }


def create_error_response(
    message: str,
    error_code: str,
    query_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    technical_details: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        message=message,
        error_details=ErrorDetails(
            error_code=error_code,
            error_type=error_code.split('_')[0],
            user_message=message,
            technical_details=technical_details,
            suggestions=suggestions or []
        ),
        query_id=query_id,
        correlation_id=correlation_id
    )


# Export all response models
__all__ = [
    'BaseResponse',
    'ChatResponse',
    'ChatData',
    'ChatMessage',
    'StreamingChatResponse',
    'SearchResponse',
    'SearchData',
    'ErrorResponse',
    'ErrorDetails',
    'HealthStatus',
    'MetricsResponse',
    'MetricsData',
    'UserStatsResponse',
    'UserStats',
    'ResponseMetadata',
    'CostPrediction',
    'CostBreakdown',
    'DeveloperHints',
    'ConversationContext',
    'OpenAIChatResponse',
    'OpenAIChoice',
    'create_success_response',
    'create_error_response'
]
