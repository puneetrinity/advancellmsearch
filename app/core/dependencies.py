"""
Centralized dependency injection module for FastAPI application.
"""
from fastapi import Request

from app.core.logging import get_logger

logger = get_logger("dependencies")

# Example: import your services, database, cache, etc.
# from app.models.manager import ModelManager
# from app.cache.redis_client import RedisClient
# from app.analytics.clickhouse_client import ClickHouseClient

# Define dependency providers here
# def get_model_manager() -> ModelManager:
#     ...
# def get_redis_client() -> RedisClient:
#     ...
# def get_clickhouse_client() -> ClickHouseClient:
#     ...

# Add more dependency providers as needed for your application

# Example placeholder dependency
def get_settings():
    # Import and return your settings/config object here
    from app.core.config import settings
    return settings

async def get_dependencies_safely(request: Request):
    """Safely get dependencies from app state with fallbacks."""
    app_state = getattr(request.app.state, 'app_state', {})
    model_manager = app_state.get("model_manager")
    cache_manager = app_state.get("cache_manager")
    chat_graph = app_state.get("chat_graph")
    search_graph = app_state.get("search_graph")
    if model_manager is None:
        logger.warning("ModelManager not available")
    if cache_manager is None:
        logger.warning("CacheManager not available")
    return {
        "model_manager": model_manager,
        "cache_manager": cache_manager,
        "chat_graph": chat_graph,
        "search_graph": search_graph
    }
