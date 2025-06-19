# app/schemas/requests.py
"""
API Request schemas
Pydantic models for request validation
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class Constraints(BaseModel):
    """Request constraints"""
    max_cost: Optional[float] = Field(0.05, description="Maximum cost in INR")
    max_time: Optional[float] = Field(5.0, description="Maximum execution time in seconds")
    quality_requirement: Optional[str] = Field("balanced", description="Quality level: minimal, balanced, high, premium")
    force_local_only: Optional[bool] = Field(False, description="Force local models only")
    
    @field_validator('quality_requirement')
    def validate_quality(cls, v):
        allowed = ["minimal", "balanced", "high", "premium"]
        if v not in allowed:
            raise ValueError(f"Quality requirement must be one of: {allowed}")
        return v


class Context(BaseModel):
    """Request context"""
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    session_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator('role')
    def validate_role(cls, v):
        allowed = ["user", "assistant", "system"]
        if v not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v


class ChatRequest(BaseModel):
    """Non-streaming chat request"""
    message: str = Field(..., description="User message", min_length=1, max_length=8000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[Context] = Field(None, description="Additional context")
    constraints: Optional[Constraints] = Field(None, description="Request constraints")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Explain how async/await works in Python",
                "session_id": "session_123",
                "context": {
                    "user_preferences": {
                        "response_style": "detailed",
                        "technical_level": "intermediate"
                    }
                },
                "constraints": {
                    "max_cost": 0.10,
                    "max_time": 10.0,
                    "quality_requirement": "high"
                }
            }
        }
    }


class ChatStreamRequest(BaseModel):
    """Streaming chat request (OpenAI-compatible)"""
    messages: List[ChatMessage] = Field(..., description="List of messages in conversation")
    session_id: Optional[str] = Field(None, description="Session ID")
    model: Optional[str] = Field("auto", description="Preferred model (auto-selects if not specified)")
    max_tokens: Optional[int] = Field(300, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    stream: bool = Field(True, description="Enable streaming")
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    quality_requirement: Optional[str] = Field("balanced")
    max_completion_time: Optional[float] = Field(30.0)
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @field_validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v < 1 or v > 4000:
            raise ValueError("Max tokens must be between 1 and 4000")
        return v


class SearchRequest(BaseModel):
    """Search analysis request"""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    search_depth: Optional[str] = Field("balanced", description="Search depth: shallow, balanced, deep, comprehensive")
    sources: Optional[List[str]] = Field(["web"], description="Sources to search: web, academic, news, technical")
    analysis_type: Optional[str] = Field("summary", description="Analysis type: summary, detailed, comparative")
    constraints: Optional[Constraints] = Field(None, description="Request constraints")
    session_id: Optional[str] = Field(None, description="Session ID")
    
    @field_validator('search_depth')
    def validate_search_depth(cls, v):
        allowed = ["shallow", "balanced", "deep", "comprehensive"]
        if v not in allowed:
            raise ValueError(f"Search depth must be one of: {allowed}")
        return v
    
    @field_validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed = ["summary", "detailed", "comparative", "analytical"]
        if v not in allowed:
            raise ValueError(f"Analysis type must be one of: {allowed}")
        return v
    
    @field_validator('sources')
    def validate_sources(cls, v):
        allowed = ["web", "academic", "news", "technical", "social"]
        if not all(source in allowed for source in v):
            raise ValueError(f"All sources must be from: {allowed}")
        return v


class ResearchRequest(BaseModel):
    """Deep research request"""
    research_question: str = Field(..., description="Research question", min_length=10, max_length=1000)
    methodology: Optional[str] = Field("systematic", description="Research methodology: systematic, exploratory, comparative")
    time_budget: Optional[int] = Field(300, description="Time budget in seconds")
    cost_budget: Optional[float] = Field(0.50, description="Cost budget in INR")
    sources: Optional[List[str]] = Field(["web", "academic"], description="Research sources")
    depth_level: Optional[int] = Field(3, description="Research depth level (1-5)")
    session_id: Optional[str] = Field(None, description="Session ID")
    
    @field_validator('methodology')
    def validate_methodology(cls, v):
        allowed = ["systematic", "exploratory", "comparative", "meta-analysis"]
        if v not in allowed:
            raise ValueError(f"Methodology must be one of: {allowed}")
        return v
    
    @field_validator('depth_level')
    def validate_depth_level(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Depth level must be between 1 and 5")
        return v


# app/schemas/responses.py
"""
API Response schemas
Standardized response formats
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class CostBreakdown(BaseModel):
    """Cost breakdown item"""
    step: str = Field(..., description="Processing step name")
    model: Optional[str] = Field(None, description="Model used")
    provider: Optional[str] = Field(None, description="Service provider")
    cost: float = Field(..., description="Cost for this step")


class CostPrediction(BaseModel):
    """Cost prediction and optimization"""
    estimated_cost: float = Field(..., description="Total estimated cost")
    cost_breakdown: List[CostBreakdown] = Field(default_factory=list)
    savings_tips: List[str] = Field(default_factory=list)
    alternative_workflows: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class DeveloperHints(BaseModel):
    """Developer debugging and optimization hints"""
    suggested_next_queries: Optional[List[str]] = Field(default_factory=list)
    potential_optimizations: Optional[Dict[str, str]] = Field(default_factory=dict)
    knowledge_gaps: Optional[List[str]] = Field(default_factory=list)
    routing_explanation: Optional[str] = Field(None)
    execution_path: Optional[List[str]] = Field(default_factory=list)
    performance: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ResponseMetadata(BaseModel):
    """Response metadata"""
    query_id: str = Field(..., description="Unique query identifier")
    execution_time: float = Field(..., description="Total execution time in seconds")
    cost: float = Field(..., description="Total cost incurred")
    models_used: List[str] = Field(default_factory=list)
    confidence: float = Field(..., description="Average confidence score")
    cached: bool = Field(False, description="Whether response was cached")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())


class BaseResponse(BaseModel):
    """Base response format"""
    status: str = Field(..., description="Response status: success, error, partial")
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[ResponseMetadata] = Field(None)
    cost_prediction: Optional[CostPrediction] = Field(None)
    developer_hints: Optional[DeveloperHints] = Field(None)


class ChatResponse(BaseResponse):
    """Chat completion response"""
    data: Dict[str, Any] = Field(..., description="Chat response data")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "data": {
                    "response": "Async/await in Python allows you to write asynchronous code...",
                    "session_id": "session_123"
                },
                "metadata": {
                    "query_id": "query_456",
                    "execution_time": 1.23,
                    "cost": 0.008,
                    "models_used": ["llama2:7b"],
                    "confidence": 0.89,
                    "cached": False
                },
                "cost_prediction": {
                    "estimated_cost": 0.008,
                    "cost_breakdown": [
                        {"step": "classification", "model": "phi:mini", "cost": 0.0},
                        {"step": "generation", "model": "llama2:7b", "cost": 0.0}
                    ],
                    "savings_tips": ["Use phi:mini for simpler questions"]
                },
                "developer_hints": {
                    "execution_path": ["context_manager", "intent_classifier", "response_generator"],
                    "routing_explanation": "Routed to code assistance based on programming keywords",
                    "performance": {
                        "cache_hits": 1,
                        "total_steps": 3,
                        "avg_confidence": 0.89
                    }
                }
            }
        }
    }


class SearchResponse(BaseResponse):
    """Search analysis response"""
    data: Dict[str, Any] = Field(..., description="Search results and analysis")


class ResearchResponse(BaseResponse):
    """Research response"""
    data: Dict[str, Any] = Field(..., description="Research findings and analysis")


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = Field("error", description="Error status")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Specific error code")
    query_id: Optional[str] = Field(None, description="Query ID if available")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "error",
                "message": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "query_id": "query_789",
                "timestamp": "2025-06-19T10:30:00Z"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="System health status")
    components: Dict[str, str] = Field(..., description="Component health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MetricsResponse(BaseModel):
    """System metrics response"""
    status: str = Field(..., description="Response status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class UsageStats(BaseModel):
    """User usage statistics"""
    total_queries: int = Field(0, description="Total queries made")
    queries_today: int = Field(0, description="Queries made today")
    total_cost: float = Field(0.0, description="Total cost incurred")
    cost_today: float = Field(0.0, description="Cost incurred today")
    remaining_budget: float = Field(0.0, description="Remaining monthly budget")
    avg_response_time: float = Field(0.0, description="Average response time")
    cache_hit_rate: float = Field(0.0, description="Cache hit rate")
    favorite_models: List[str] = Field(default_factory=list)


class UserStatsResponse(BaseModel):
    """User statistics response"""
    status: str = Field("success")
    data: UsageStats = Field(..., description="User usage statistics")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Utility response models

class ListResponse(BaseModel):
    """Generic list response"""
    status: str = Field("success")
    data: Dict[str, Any] = Field(..., description="List data")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination info")
    total: int = Field(0, description="Total items")


class StatusResponse(BaseModel):
    """Simple status response"""
    status: str = Field(..., description="Operation status")
    message: Optional[str] = Field(None, description="Status message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")
