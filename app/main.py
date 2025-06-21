# app/main.py
"""
Production-ready main application with comprehensive initialization and monitoring.
Integrates all components for the complete AI search system with standardized providers.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

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
from app.graphs.search_graph import SearchGraph
from app.performance.optimization import OptimizedSearchSystem
from app.schemas.responses import HealthStatus, create_error_response


# Global state for application components
app_state: Dict[str, Any] = {}
settings = get_settings()
logger = get_logger("main")


from app.graphs.search_graph import execute_search

class SearchSystemWrapper:
    def __init__(self, model_manager, cache_manager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
    async def search(self, query, budget=2.0, quality="balanced", max_results=10, **kwargs):
        return await execute_search(
            query=query,
            model_manager=self.model_manager,
            cache_manager=self.cache_manager,
            budget=budget,
            quality=quality,
            max_results=max_results
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    Handles startup and shutdown procedures for the standardized architecture.
    """
    # Startup
    logger.info("🚀 Starting AI Search System with Standardized Providers")
    
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
        
        # Initialize search graph with standardized providers
        logger.info("🔍 Initializing search graph with Brave + ScrapingBee...")
        search_graph = SearchGraph(model_manager, app_state["cache_manager"])
        app_state["search_graph"] = search_graph
        logger.info("✅ Search graph initialized")
        
        # Initialize optimization system for search-augmented chat
        logger.info("⚡ Initializing optimization system...")
        search_router = SearchSystemWrapper(model_manager, app_state["cache_manager"])
        search_system = OptimizedSearchSystem(
            search_router=search_router,
            search_graph=search_graph
        )
        app_state["search_system"] = search_system
        logger.info("✅ Optimization system initialized")
        
        # Validate provider configuration
        logger.info("🔑 Validating provider API keys...")
        api_key_status = {
            "brave_search": bool(getattr(settings, 'brave_search_api_key', None) or 
                                getattr(settings, 'BRAVE_API_KEY', None) or 
                                os.getenv('BRAVE_API_KEY')),
            "scrapingbee": bool(getattr(settings, 'scrapingbee_api_key', None) or 
                               getattr(settings, 'SCRAPINGBEE_API_KEY', None) or 
                               os.getenv('SCRAPINGBEE_API_KEY'))
        }
        
        app_state["api_key_status"] = api_key_status
        
        if api_key_status["brave_search"]:
            logger.info("✅ Brave Search API key configured")
        else:
            logger.warning("⚠️  Brave Search API key not found - search functionality will be limited")
        
        if api_key_status["scrapingbee"]:
            logger.info("✅ ScrapingBee API key configured")
        else:
            logger.warning("⚠️  ScrapingBee API key not found - content enhancement disabled")
        
        # Set dependencies for API modules
        chat.set_dependencies(model_manager, app_state["cache_manager"], chat_graph)
        search.set_dependencies(model_manager, app_state["cache_manager"], search_graph)
        logger.info("✅ API dependencies configured")
        
        # Application startup complete
        startup_time = time.time()
        app_state["startup_time"] = startup_time
        
        logger.info("🎉 AI Search System startup completed successfully")
        logger.info(f"📊 Components initialized: {len(app_state)} components")
        logger.info(f"🏗️  Architecture: Standardized Providers (Brave + ScrapingBee)")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}", exc_info=True)
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down AI Search System...")
    
    try:
        # Cleanup search graph (and its providers)
        if "search_graph" in app_state:
            try:
                await app_state["search_graph"].cleanup()
                logger.info("✅ Search graph and providers cleaned up")
            except Exception as e:
                logger.warning(f"⚠️  Search graph cleanup error: {e}")
        
        # Cleanup model manager
        if "model_manager" in app_state:
            await app_state["model_manager"].cleanup()
            logger.info("✅ Model manager cleaned up")
        
        # Cleanup cache manager
        if "cache_manager" in app_state and app_state["cache_manager"]:
            try:
                await app_state["cache_manager"].close()
                logger.info("✅ Cache manager cleaned up")
            except Exception as e:
                logger.warning(f"⚠️  Cache cleanup error: {e}")
        
        logger.info("✅ AI Search System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AI Search System",
    description="Intelligent AI-powered search with Brave Search and ScrapingBee integration",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
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


@app.middleware("http")
async def app_state_middleware(request: Request, call_next):
    """Ensure app.state has access to components."""
    if hasattr(request.app, 'state'):
        request.app.state.search_system = app_state.get("search_system")
        request.app.state.model_manager = app_state.get("model_manager")
        request.app.state.cache_manager = app_state.get("cache_manager")
        request.app.state.chat_graph = app_state.get("chat_graph")
        request.app.state.search_graph = app_state.get("search_graph")
    response = await call_next(request)
    return response


# Performance monitoring middleware
@app.middleware("http")
async def performance_tracking_middleware(request: Request, call_next):
    """Middleware to track request performance."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        response_time = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Response-Time"] = str(round(response_time * 1000, 2))
        response.headers["X-Request-ID"] = get_correlation_id()
        
        # Log slow requests
        if response_time > 5.0:  # 5 seconds
            logger.warning(
                "Slow request detected",
                method=request.method,
                url=str(request.url),
                response_time=response_time,
                status_code=response.status_code
            )
        
        return response
    
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            response_time=response_time,
            error=str(e)
        )
        raise


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
        
        # Check search graph
        if "search_graph" in app_state:
            components["search_graph"] = "healthy"
        else:
            components["search_graph"] = "not_initialized"
            overall_healthy = False
        
        # Check optimization system
        if "search_system" in app_state:
            components["optimization_system"] = "healthy"
        else:
            components["optimization_system"] = "not_initialized"
            overall_healthy = False
        
        # Check provider API keys
        api_key_status = app_state.get("api_key_status", {})
        if api_key_status.get("brave_search", False):
            components["brave_search"] = "configured"
        else:
            components["brave_search"] = "not_configured"
            # Don't mark as unhealthy since system can work without it
        
        if api_key_status.get("scrapingbee", False):
            components["scrapingbee"] = "configured"
        else:
            components["scrapingbee"] = "not_configured"
            # Don't mark as unhealthy since it's optional
        
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
    required_components = ["model_manager", "chat_graph", "search_graph"]
    
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


@app.get("/system/status")
async def system_status():
    """Detailed system status endpoint."""
    try:
        # Component status
        redis_status = "disconnected"
        ollama_status = "disconnected"
        
        if "cache_manager" in app_state and app_state["cache_manager"]:
            redis_status = "connected"
        
        if "model_manager" in app_state:
            try:
                stats = app_state["model_manager"].get_model_stats()
                ollama_status = "connected"
            except:
                ollama_status = "error"
        
        # Provider status
        api_key_status = app_state.get("api_key_status", {})
        
        # Calculate uptime
        uptime = None
        if "startup_time" in app_state:
            uptime = time.time() - app_state["startup_time"]
        
        return {
            "status": "operational",
            "components": {
                "redis": redis_status,
                "ollama": ollama_status,
                "api": "healthy",
                "search_graph": "initialized" if "search_graph" in app_state else "not_initialized",
                "chat_graph": "initialized" if "chat_graph" in app_state else "not_initialized"
            },
            "providers": {
                "brave_search": "configured" if api_key_status.get("brave_search", False) else "not_configured",
                "scrapingbee": "configured" if api_key_status.get("scrapingbee", False) else "not_configured"
            },
            "version": "1.0.0",
            "uptime": uptime,
            "timestamp": time.time(),
            "architecture": "standardized_providers"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring with debug for circular/self-references."""
    import json
    try:
        metrics = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
        }
        # Debug: Model stats
        model_stats = None
        try:
            model_stats = app_state["model_manager"].get_model_stats()
            print("✅ Model stats OK")
        except Exception as e:
            print(f"❌ Model stats error: {e}")
            model_stats = {"error": "model_stats_failed"}
        try:
            json.dumps(model_stats)
            print("✅ Model stats JSON serializable")
        except Exception as e:
            print(f"❌ Model stats not serializable: {e}")
        metrics["models"] = model_stats
        # Debug: Chat graph stats
        graph_stats = None
        try:
            graph_stats = app_state["chat_graph"].get_performance_stats()
            print("✅ Graph stats OK")
        except Exception as e:
            print(f"❌ Graph stats error: {e}")
            graph_stats = {"error": "graph_stats_failed"}
        try:
            json.dumps(graph_stats)
            print("✅ Graph stats JSON serializable")
        except Exception as e:
            print(f"❌ Graph stats not serializable: {e}")
        metrics["chat_graph"] = graph_stats
        # Add provider availability
        api_key_status = app_state.get("api_key_status", {})
        metrics["providers"] = {
            "brave_search_available": api_key_status.get("brave_search", False),
            "scrapingbee_available": api_key_status.get("scrapingbee", False)
        }
        return metrics
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
    api_key_status = app_state.get("api_key_status", {})
    
    return {
        "name": "AI Search System",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.environment,
        "architecture": "standardized_providers",
        "docs_url": "/docs" if settings.environment != "production" else None,
        "health_url": "/health",
        "api_endpoints": {
            "chat": "/api/v1/chat/complete",
            "chat_stream": "/api/v1/chat/stream",
            "search_basic": "/api/v1/search/basic",
            "search_advanced": "/api/v1/search/advanced",
            "health": "/health",
            "metrics": "/metrics",
            "system_status": "/system/status"
        },
        "features": [
            "Intelligent conversation management",
            "Multi-model AI orchestration", 
            "Real-time streaming responses",
            "Brave Search integration",
            "ScrapingBee content enhancement",
            "Smart cost optimization",
            "Context-aware processing",
            "Performance monitoring"
        ],
        "providers": {
            "brave_search": "configured" if api_key_status.get("brave_search", False) else "not_configured",
            "scrapingbee": "configured" if api_key_status.get("scrapingbee", False) else "not_configured",
            "local_models": "ollama"
        }
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
                    "connected": value is not None
                }
            elif key == "chat_graph":
                debug_state[key] = {
                    "type": "ChatGraph",
                    "initialized": True
                }
            elif key == "search_graph":
                debug_state[key] = {
                    "type": "SearchGraph", 
                    "initialized": True,
                    "providers": "brave_search + scrapingbee"
                }
            elif key == "api_key_status":
                debug_state[key] = value
            else:
                debug_state[key] = str(type(value))
        
        return {
            "application_state": debug_state,
            "settings": {
                "environment": settings.environment,
                "debug": settings.debug,
                "ollama_host": settings.ollama_host,
                "redis_url": settings.redis_url.split('@')[-1] if '@' in settings.redis_url else settings.redis_url,  # Hide credentials
                "log_level": settings.log_level
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
                "response": getattr(result, 'final_response', 'No response generated'),
                "execution_time": getattr(result, 'execution_time', 0),
                "cost": result.calculate_total_cost() if hasattr(result, 'calculate_total_cost') else 0,
                "execution_path": getattr(result, 'execution_path', [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @app.post("/debug/test-search")
    async def debug_test_search(query: str = "latest AI developments"):
        """Debug endpoint to test search functionality."""
        try:
            if "search_graph" not in app_state:
                return {"error": "Search graph not initialized"}
            
            from app.graphs.search_graph import execute_search
            
            result = await execute_search(
                query=query,
                model_manager=app_state["model_manager"],
                cache_manager=app_state["cache_manager"],
                budget=2.0,
                quality="balanced",
                max_results=5
            )
            
            return {
                "success": result.get("success", False),
                "query": query,
                "response": result.get("response", "No response"),
                "metadata": result.get("metadata", {}),
                "providers_used": result.get("metadata", {}).get("providers_used", [])
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
    print(f"🏗️  Architecture: Standardized Providers (Brave + ScrapingBee)")
    print(f"📍 Server will be available at http://{settings.api_host}:{settings.api_port}")
    print(f"📚 API documentation at http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🏥 Health check at http://{settings.api_host}:{settings.api_port}/health")
    print(f"📊 System status at http://{settings.api_host}:{settings.api_port}/system/status")
    
    # API key status
    brave_key = (getattr(settings, 'brave_search_api_key', None) or 
                getattr(settings, 'BRAVE_API_KEY', None) or 
                os.getenv('BRAVE_API_KEY'))
    scrapingbee_key = (getattr(settings, 'scrapingbee_api_key', None) or 
                      getattr(settings, 'SCRAPINGBEE_API_KEY', None) or 
                      os.getenv('SCRAPINGBEE_API_KEY'))
    
    print(f"🔑 Brave Search: {'✅ Configured' if brave_key else '❌ Not configured'}")
    print(f"🔑 ScrapingBee: {'✅ Configured' if scrapingbee_key else '❌ Not configured'}")
    
    if not brave_key:
        print("⚠️  Warning: No Brave Search API key found. Set BRAVE_API_KEY environment variable.")
    if not scrapingbee_key:
        print("⚠️  Warning: No ScrapingBee API key found. Set SCRAPINGBEE_API_KEY environment variable.")
    
    # Run the server
    uvicorn.run("app.main:app", **dev_config)


# Production ASGI application
def get_asgi_application():
    """Get ASGI application for production deployment."""
    return app


# Export for testing
__all__ = [
    'app', 
    'create_app', 
    'get_asgi_application',
    'app_state'
]


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


# Kubernetes deployment configuration
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


"""
Production deployment configuration:

For Gunicorn (gunicorn.conf.py):
```python
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

# Process naming
proc_name = "ai-search-system"

# Worker tuning
worker_tmp_dir = "/dev/shm"
```

Run with: gunicorn -c gunicorn.conf.py app.main:app

Environment Variables Required:
- BRAVE_API_KEY=your_brave_search_api_key
- SCRAPINGBEE_API_KEY=your_scrapingbee_api_key  
- REDIS_URL=redis://localhost:6379
- OLLAMA_HOST=http://localhost:11434
"""