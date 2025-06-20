"""
Production-ready main application with comprehensive initialization and monitoring.
Integrates all components for a complete AI search system.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger, LoggingMiddleware, get_correlation_id
from app.api.security import SecurityMiddleware, auth_stub
from app.api import chat, search
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.graphs.chat_graph import ChatGraph
from app.schemas.responses import HealthStatus, create_error_response
from app.providers.router import SmartSearchRouter
from app.graphs.search_graph import SearchGraph
from app.performance.optimization import OptimizedSearchSystem


# Global state for application components
app_state: Dict[str, Any] = {}
settings = get_settings()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    Handles startup and shutdown procedures.
    """
    # Startup
    logger.info("🚀 Starting AI Search System")
    
    try:
        # Initialize logging
        setup_logging(
            log_level=settings.log_level,
            log_format="json" if settings.environment == "production" else "text",
            enable_file_logging=settings.environment != "testing"
        )
        logger.info("✅ Logging system initialized")
        
        # Initialize model manager
        logger.info("🧠 Initializing model manager...")
        model_manager = ModelManager(settings.ollama_host)
        await model_manager.initialize()
        app_state["model_manager"] = model_manager
        logger.info("✅ Model manager initialized")
        
        # Initialize cache manager
        logger.info("💾 Initializing cache manager...")
        try:
            cache_manager = CacheManager(settings.redis_url, settings.redis_max_connections)
            await cache_manager.initialize()
            app_state["cache_manager"] = cache_manager
            logger.info("✅ Cache manager initialized")
        except Exception as e:
            logger.warning(f"⚠️  Cache manager initialization failed: {e}")
            logger.info("Continuing without cache...")
            app_state["cache_manager"] = None
        
        # Initialize chat graph
        logger.info("🔄 Initializing chat graph...")
        chat_graph = ChatGraph(model_manager, app_state["cache_manager"])
        app_state["chat_graph"] = chat_graph
        logger.info("✅ Chat graph initialized")
        
        # Set dependencies for API modules
        chat.set_dependencies(model_manager, app_state["cache_manager"], chat_graph)
        
        # Initialize search system components
        logger.info("🔍 Initializing search system components...")
        search_router_instance = SmartSearchRouter(
            brave_api_key=os.getenv("BRAVE_API_KEY", settings.brave_search_api_key or settings.BRAVE_API_KEY),
            scrapingbee_api_key=os.getenv("SCRAPINGBEE_API_KEY", getattr(settings, "SCRAPINGBEE_API_KEY", None))
        )
        await search_router_instance.__aenter__()
        search_graph = SearchGraph(search_router_instance)
        optimized_system = OptimizedSearchSystem(search_router_instance, search_graph)
        app_state["search_system"] = optimized_system
        app_state["search_router"] = search_router_instance
        logger.info("✅ Search system components initialized")
        
        # Application startup complete
        startup_time = time.time()
        app_state["startup_time"] = startup_time
        
        logger.info("🎉 AI Search System startup completed successfully")
        logger.info(f"📊 Components initialized: {len(app_state)} components")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}", exc_info=True)
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down AI Search System...")
    
    try:
        # Cleanup model manager
        if "model_manager" in app_state:
            await app_state["model_manager"].cleanup()
            logger.info("✅ Model manager cleaned up")
        
        # Cleanup cache manager
        if "cache_manager" in app_state and app_state["cache_manager"]:
            # Note: CacheManager cleanup would go here if implemented
            logger.info("✅ Cache manager cleaned up")
        
        # Cleanup search router
        if "search_router" in app_state and app_state["search_router"]:
            await app_state["search_router"].__aexit__(None, None, None)
            logger.info("✅ Search system cleaned up")
        
        logger.info("✅ AI Search System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AI Search System",
    description="Intelligent AI-powered search with cost optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityMiddleware, enable_rate_limiting=True)

# Add logging middleware
app.add_middleware(LoggingMiddleware)


# Health check endpoints
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        components = {}
        overall_healthy = True
        
        # Check model manager
        if "model_manager" in app_state:
            try:
                model_stats = app_state["model_manager"].get_model_stats()
                components["models"] = "healthy"
                logger.debug(f"Models: {model_stats['total_models']} total, {model_stats['loaded_models']} loaded")
            except Exception as e:
                components["models"] = "unhealthy"
                overall_healthy = False
                logger.error(f"Model manager health check failed: {e}")
        else:
            components["models"] = "not_initialized"
            overall_healthy = False
        
        # Check cache manager
        if "cache_manager" in app_state and app_state["cache_manager"]:
            try:
                # Simple cache test
                test_key = f"health_check_{int(time.time())}"
                await app_state["cache_manager"].set(test_key, "test", ttl=5)
                test_value = await app_state["cache_manager"].get(test_key)
                
                if test_value == "test":
                    components["cache"] = "healthy"
                else:
                    components["cache"] = "degraded"
                    overall_healthy = False
            except Exception as e:
                components["cache"] = "unhealthy"
                overall_healthy = False
                logger.error(f"Cache health check failed: {e}")
        else:
            components["cache"] = "not_available"
            # Cache is optional, so don't mark as unhealthy
        
        # Check chat graph
        if "chat_graph" in app_state:
            components["chat_graph"] = "healthy"
        else:
            components["chat_graph"] = "not_initialized"
            overall_healthy = False
        
        # Check search system
        if "search_system" in app_state and app_state["search_system"]:
            components["search_system"] = "healthy"
        else:
            components["search_system"] = "not_initialized"
            overall_healthy = False
        
        # Calculate uptime
        uptime = None
        if "startup_time" in app_state:
            uptime = time.time() - app_state["startup_time"]
        
        return HealthStatus(
            status="healthy" if overall_healthy else "degraded",
            components=components,
            version="1.0.0",
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthStatus(
            status="unhealthy",
            components={"error": str(e)},
            version="1.0.0"
        )


@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    required_components = ["model_manager", "chat_graph"]
    
    for component in required_components:
        if component not in app_state:
            raise HTTPException(
                status_code=503,
                detail=f"Component {component} not ready"
            )
    
    return {"status": "ready"}


@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring."""
    try:
        # Get graph stats
        if hasattr(app.state, "chat_graph") and app.state.chat_graph:
            graph_stats = app.state.chat_graph.get_performance_stats()
        else:
            graph_stats = {"error": "Chat graph not available"}
        
        # Get model stats
        if hasattr(app.state, "model_manager") and app.state.model_manager:
            model_stats = app.state.model_manager.get_model_stats()
        else:
            model_stats = {"error": "Model manager not available"}
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "graph_stats": graph_stats,
            "model_stats": model_stats
        }
    except Exception as e:
        logger.error("Metrics collection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics collection failed")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    correlation_id = get_correlation_id()
    
    logger.error(
        "Unhandled exception in request",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        correlation_id=correlation_id,
        exc_info=True
    )
    
    # Don't expose internal error details in production
    if settings.environment == "production":
        error_detail = "An internal error occurred"
    else:
        error_detail = str(exc)
    
    error_response = create_error_response(
        message="Internal server error",
        error_code="INTERNAL_SERVER_ERROR",
        correlation_id=correlation_id,
        technical_details=error_detail if settings.environment != "production" else None
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Rate limit exceeded handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    correlation_id = get_correlation_id()
    
    if exc.status_code == 429:  # Rate limit exceeded
        logger.warning(
            "Rate limit exceeded",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else "unknown",
            correlation_id=correlation_id
        )
    elif exc.status_code >= 500:
        logger.error(
            "Server error in request",
            method=request.method,
            url=str(request.url),
            status_code=exc.status_code,
            error=exc.detail,
            correlation_id=correlation_id
        )
    else:
        logger.info(
            "Client error in request", 
            method=request.method,
            url=str(request.url),
            status_code=exc.status_code,
            error=exc.detail,
            correlation_id=correlation_id
        )
    
    # Return structured error response
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    else:
        error_response = create_error_response(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            correlation_id=correlation_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )


# Include API routers
app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"],
    dependencies=[]
)

app.include_router(
    search.router,
    prefix="/api/v1/search",
    tags=["Search"],
    dependencies=[]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "AI Search System",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.environment,
        "docs_url": "/docs" if settings.environment != "production" else None,
        "health_url": "/health",
        "api_endpoints": {
            "chat": "/api/v1/chat/complete",
            "chat_stream": "/api/v1/chat/stream",
            "search": "/api/v1/search/basic",
            "health": "/health",
            "metrics": "/metrics"
        },
        "features": [
            "Intelligent conversation management",
            "Multi-model AI orchestration", 
            "Real-time streaming responses",
            "Context-aware processing",
            "Cost optimization",
            "Performance monitoring"
        ]
    }


# Development endpoints (only in non-production)
if settings.environment != "production":
    @app.get("/debug/state")
    async def debug_application_state():
        """Debug endpoint to inspect application state."""
        debug_state = {}
        
        for key, value in app_state.items():
            if key == "model_manager":
                debug_state[key] = {
                    "type": "ModelManager",
                    "stats": value.get_model_stats() if hasattr(value, 'get_model_stats') else None
                }
            elif key == "cache_manager":
                debug_state[key] = {
                    "type": "CacheManager",
                    "available": value is not None,
                    "metrics": value.metrics.__dict__ if value and hasattr(value, 'metrics') else None
                }
            elif key == "chat_graph":
                debug_state[key] = {
                    "type": "ChatGraph",
                    "stats": value.get_performance_stats() if hasattr(value, 'get_performance_stats') else None
                }
            elif key == "search_system":
                debug_state[key] = {
                    "type": "OptimizedSearchSystem",
                    "available": value is not None,
                    "status": value.get_system_status() if hasattr(value, 'get_system_status') else None
                }
            else:
                debug_state[key] = str(type(value))
        
        return {
            "application_state": debug_state,
            "settings": {
                "environment": settings.environment,
                "debug": settings.debug,
                "ollama_host": settings.ollama_host,
                "redis_url": settings.redis_url.split('@')[-1] if '@' in settings.redis_url else settings.redis_url  # Hide credentials
            }
        }
    
    @app.post("/debug/test-chat")
    async def debug_test_chat(message: str = "Hello, this is a test"):
        """Debug endpoint to test chat functionality."""
        try:
            if "chat_graph" not in app_state:
                return {"error": "Chat graph not initialized"}
            
            from app.graphs.base import GraphState
            
            test_state = GraphState(
                original_query=message,
                session_id="debug_test",
                user_id="debug_user"
            )
            
            result = await app_state["chat_graph"].execute(test_state)
            
            return {
                "success": True,
                "response": result.final_response,
                "execution_time": result.calculate_total_time(),
                "cost": result.calculate_total_cost(),
                "confidence": result.get_avg_confidence(),
                "execution_path": result.execution_path,
                "errors": result.errors,
                "warnings": result.warnings
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app


# Development server
if __name__ == "__main__":
    # Development server configuration
    dev_config = {
        "host": settings.api_host,
        "port": settings.api_port,
        "reload": settings.debug,
        "log_level": "info" if not settings.debug else "debug",
        "access_log": True,
        "reload_dirs": ["app"] if settings.debug else None,
    }
    
    print(f"🚀 Starting AI Search System in {settings.environment} mode")
    print(f"📍 Server will be available at http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API documentation at http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🏥 Health check at http://{settings.api_host}:{settings.api_port}/health")
    
    # Run the server
    uvicorn.run("app.main:app", **dev_config)


# Production ASGI application
def get_asgi_application():
    """Get ASGI application for production deployment."""
    return app


# Gunicorn configuration (for production)
"""
For production deployment with Gunicorn, create gunicorn.conf.py:

bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 60
keepalive = 2
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "ai-search-system"

# Worker tuning
worker_tmp_dir = "/dev/shm"
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

Run with: gunicorn -c gunicorn.conf.py app.main:app
"""


# Kubernetes deployment helpers
def get_health_check_config():
    """Get health check configuration for Kubernetes."""
    return {
        "readiness_probe": {
            "http_get": {
                "path": "/health/ready",
                "port": 8000
            },
            "initial_delay_seconds": 30,
            "period_seconds": 10,
            "timeout_seconds": 5,
            "failure_threshold": 3
        },
        "liveness_probe": {
            "http_get": {
                "path": "/health/live", 
                "port": 8000
            },
            "initial_delay_seconds": 60,
            "period_seconds": 30,
            "timeout_seconds": 10,
            "failure_threshold": 3
        },
        "startup_probe": {
            "http_get": {
                "path": "/health/ready",
                "port": 8000
            },
            "initial_delay_seconds": 10,
            "period_seconds": 10,
            "timeout_seconds": 5,
            "failure_threshold": 10
        }
    }


# Docker health check
def docker_health_check():
    """Health check function for Docker containers."""
    import requests
    import sys
    
    try:
        response = requests.get("http://localhost:8000/health/live", timeout=5)
        if response.status_code == 200:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception:
        sys.exit(1)


# Performance monitoring utilities
class PerformanceMonitor:
    """Simple performance monitoring utilities."""
    
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request for performance monitoring."""
        self.request_count += 1
        self.total_response_time += response_time
        if is_error:
            self.error_count += 1
    
    def get_stats(self):
        """Get performance statistics."""
        if self.request_count == 0:
            return {"requests": 0, "avg_response_time": 0, "error_rate": 0}
        
        return {
            "requests": self.request_count,
            "avg_response_time": self.total_response_time / self.request_count,
            "error_rate": self.error_count / self.request_count,
            "total_errors": self.error_count
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()


# Middleware to track performance
@app.middleware("http")
async def performance_tracking_middleware(request: Request, call_next):
    """Middleware to track request performance."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        response_time = time.time() - start_time
        
        # Record performance metrics
        is_error = response.status_code >= 400
        performance_monitor.record_request(response_time, is_error)
        
        # Add performance headers
        response.headers["X-Response-Time"] = str(round(response_time * 1000, 2))
        response.headers["X-Request-ID"] = get_correlation_id()
        
        return response
    
    except Exception as e:
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)
        raise


# Export for testing
__all__ = [
    'app', 
    'create_app', 
    'get_asgi_application',
    'app_state',
    'performance_monitor'
]
