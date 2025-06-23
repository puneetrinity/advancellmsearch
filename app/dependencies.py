"""
Dependency providers for FastAPI DI: ModelManager and CacheManager singletons.
"""
from functools import lru_cache
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.core.config import get_settings

@lru_cache()
def get_model_manager() -> ModelManager:
    settings = get_settings()
    return ModelManager(ollama_host=settings.ollama_host)

@lru_cache()
def get_cache_manager() -> CacheManager:
    settings = get_settings()
    return CacheManager(redis_url=settings.redis_url, max_connections=settings.redis_max_connections)
