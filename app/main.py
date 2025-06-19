# app/main.py
"""
AI Search System - Main FastAPI Application
Intelligence lives in APIs, not interfaces.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.chat import router as chat_router
from app.api.search import router as search_router
from app.api.middleware import RateLimitMiddleware, CostTrackingMiddleware
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager

# Configure structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Global instances
model_manager: ModelManager = None
cache_manager: CacheManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    settings = get_settings()
    
    logger.info("🚀 Starting AI Search System")
    
    try:
        # Initialize core components
        global model_manager, cache_manager
        
        # Initialize cache manager
        cache_manager = CacheManager(
            redis_url=settings.redis_url,
            max_connections=settings.redis_max_connections
        )
        await cache_manager.initialize()
        logger.info("✅ Cache manager initialized")
        
        # Initialize model manager
        model_manager = ModelManager(
            ollama_host=settings.ollama_host,
            cache_manager=cache_manager
        )
        await model_manager.initialize()
        logger.info("✅ Model manager initialized")
        
        # Pre-load critical models
        await model_manager.preload_models(["phi:mini"])
        logger.info("✅ Critical models preloaded")
        
        # Make managers available to routes
        app.state.model_manager = model_manager
        app.state.cache_manager = cache_manager
        
        logger.info("🎯 AI Search System ready for requests")
        
        yield
        
    except Exception as e:
        logger.error("❌ Failed to initialize system", error=str(e))
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down AI Search System")
        
        if model_manager:
            await model_manager.cleanup()
            logger.info("✅ Model manager cleaned up")
            
        if cache_manager:
            await cache_manager.cleanup()
            logger.info("✅ Cache manager cleaned up")


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Search System",
        description="Revolutionary AI search where intelligence lives in APIs",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(CostTrackingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Add routers
    app.include_router(
        chat_router,
        prefix="/api/v1/chat",
        tags=["chat"]
    )
    
    app.include_router(
        search_router,
        prefix="/api/v1/search",
        tags=["search"]
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=exc
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "query_id": getattr(request.state, "query_id", None)
            }
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """System health check"""
        try:
            # Check cache connectivity
            cache_healthy = await cache_manager.health_check() if cache_manager else False
            
            # Check model manager
            models_healthy = model_manager.is_healthy() if model_manager else False
            
            status = "healthy" if cache_healthy and models_healthy else "degraded"
            
            return {
                "status": status,
                "components": {
                    "cache": "healthy" if cache_healthy else "unhealthy",
                    "models": "healthy" if models_healthy else "unhealthy"
                },
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    # System metrics endpoint
    @app.get("/metrics")
    async def system_metrics():
        """Basic system metrics"""
        try:
            metrics = {}
            
            if cache_manager:
                metrics["cache"] = await cache_manager.get_metrics()
            
            if model_manager:
                metrics["models"] = await model_manager.get_metrics()
            
            return {
                "status": "success",
                "metrics": metrics,
                "timestamp": "2025-06-19T00:00:00Z"  # Use actual timestamp
            }
            
        except Exception as e:
            logger.error("Metrics collection failed", error=str(e))
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to collect metrics"
                }
            )
    
    return app


# Create the application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
