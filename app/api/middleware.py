# app/api/middleware.py
"""
API Middleware for authentication, rate limiting, and cost tracking
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from app.cache.redis_client import CacheManager
from app.core.config import RATE_LIMITS

logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting"""
        start_time = time.time()

        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
            response = await call_next(request)
            return response

        # Add query ID to request state
        query_id = str(uuid.uuid4())
        request.state.query_id = query_id

        try:
            response = await call_next(request)

            # Add performance headers
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Query-ID"] = query_id

            return response

        except Exception as e:
            # Log error with query ID
            logger.error(
                "Request processing error",
                query_id=query_id,
                path=request.url.path,
                error=str(e),
                exc_info=e,
            )
            raise


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Cost tracking middleware"""

    async def dispatch(self, request: Request, call_next):
        """Track costs per request"""
        # Skip for non-API endpoints
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        request.state.start_time = time.time()
        request.state.estimated_cost = 0.0

        response = await call_next(request)

        # Add cost headers if available
        if hasattr(request.state, "actual_cost"):
            response.headers["X-Request-Cost"] = str(request.state.actual_cost)

        return response


# Authentication functions


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None,
) -> Dict[str, Any]:
    """Get current user from token or create anonymous user"""

    # For development, create a mock user
    # In production, this would validate JWT tokens
    if credentials:
        # TODO: Implement JWT token validation
        # For now, extract user ID from token
        try:
            # Mock implementation - in production, decode JWT
            user_id = "user_" + credentials.credentials[-8:]  # Use last 8 chars as ID
            tier = "pro"  # Default tier
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    else:
        # Anonymous user for development
        user_id = f"anon_{uuid.uuid4().hex[:8]}"
        tier = "free"

    return {
        "id": user_id,
        "tier": tier,
        "authenticated": bool(credentials),
        "permissions": ["chat", "search"] if tier != "free" else ["chat"],
    }


async def check_rate_limit(
    user_id: str, cache_manager: CacheManager, endpoint: str = "general"
) -> bool:
    """Check if user is within rate limits"""

    # Get user tier (this would come from user data in production)
    # For now, assume based on user ID
    if user_id.startswith("anon_"):
        tier = "free"
    elif user_id.startswith("user_"):
        tier = "pro"
    else:
        tier = "enterprise"

    # Get rate limit for tier
    rate_limit = RATE_LIMITS[tier]["requests_per_minute"]

    # Check rate limit
    allowed, current_count = await cache_manager.check_rate_limit(user_id, rate_limit)

    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            user_id=user_id,
            tier=tier,
            current_count=current_count,
            limit=rate_limit,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": rate_limit,
                "current": current_count,
                "reset_in": 60,  # seconds
            },
        )

    return True


async def track_cost(user_id: str, cost: float, cache_manager: CacheManager) -> float:
    """Track and deduct cost from user budget"""

    if cost <= 0:
        return await cache_manager.get_remaining_budget(user_id)

    # Deduct from budget
    remaining_budget = await cache_manager.deduct_budget(user_id, cost)

    logger.info(
        "Cost tracked", user_id=user_id, cost=cost, remaining_budget=remaining_budget
    )

    # Warn if budget is low
    if remaining_budget < 1.0:  # Less than ₹1 remaining
        logger.warning(
            "User budget low", user_id=user_id, remaining_budget=remaining_budget
        )

    return remaining_budget


# Permission checking


def require_permission(permission: str):
    """Decorator to require specific permission"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from kwargs (should be injected by dependency)
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            if permission not in current_user.get("permissions", []):
                raise HTTPException(
                    status_code=403, detail=f"Permission '{permission}' required"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_tier(min_tier: str):
    """Decorator to require minimum user tier"""
    tier_order = ["free", "pro", "enterprise"]

    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_tier = current_user.get("tier", "free")

            if tier_order.index(user_tier) < tier_order.index(min_tier):
                raise HTTPException(
                    status_code=403, detail=f"Tier '{min_tier}' or higher required"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Budget checking


async def check_budget(
    user_id: str, estimated_cost: float, cache_manager: CacheManager
) -> bool:
    """Check if user has sufficient budget"""

    remaining_budget = await cache_manager.get_remaining_budget(user_id)

    if remaining_budget < estimated_cost:
        raise HTTPException(
            status_code=402,  # Payment Required
            detail={
                "error": "Insufficient budget",
                "remaining": remaining_budget,
                "required": estimated_cost,
                "message": "Please upgrade your plan or wait for budget reset",
            },
        )

    return True


# Request validation


async def validate_request_size(request: Request):
    """Validate request payload size"""
    content_length = request.headers.get("content-length")

    if content_length:
        size = int(content_length)
        max_size = 1024 * 1024  # 1MB limit

        if size > max_size:
            raise HTTPException(status_code=413, detail="Request payload too large")


# Security headers


async def add_security_headers(response: Response):
    """Add security headers to response"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"


# Error handling utilities


class APIError(Exception):
    """Base API error"""

    def __init__(self, message: str, status_code: int = 500, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class RateLimitError(APIError):
    """Rate limit exceeded error"""

    def __init__(self, limit: int, current: int):
        super().__init__(
            f"Rate limit exceeded: {current}/{limit} requests per minute",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class BudgetExceededError(APIError):
    """Budget exceeded error"""

    def __init__(self, required: float, remaining: float):
        super().__init__(
            f"Insufficient budget: required ₹{required:.3f}, remaining ₹{remaining:.3f}",
            status_code=402,
            error_code="BUDGET_EXCEEDED",
        )


class ModelUnavailableError(APIError):
    """Model unavailable error"""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' is currently unavailable",
            status_code=503,
            error_code="MODEL_UNAVAILABLE",
        )


# Logging utilities


def log_request_start(request: Request, user_id: str):
    """Log request start"""
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        user_id=user_id,
        query_id=getattr(request.state, "query_id", "unknown"),
        user_agent=request.headers.get("user-agent", "unknown"),
    )


def log_request_end(request: Request, response: Response, user_id: str):
    """Log request completion"""
    process_time = getattr(request.state, "process_time", 0)

    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        user_id=user_id,
        query_id=getattr(request.state, "query_id", "unknown"),
        status_code=response.status_code,
        process_time=process_time,
        cost=getattr(request.state, "actual_cost", 0),
    )


# Development utilities


async def create_dev_user(user_id: str = None) -> Dict[str, Any]:
    """Create a development user for testing"""
    if not user_id:
        user_id = f"dev_{uuid.uuid4().hex[:8]}"

    return {
        "id": user_id,
        "tier": "pro",
        "authenticated": True,
        "permissions": ["chat", "search", "research"],
        "budget": 100.0,  # ₹100 for development
        "created_at": datetime.now().isoformat(),
    }
