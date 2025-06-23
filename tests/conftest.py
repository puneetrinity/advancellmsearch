# tests/conftest.py
"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import os
from dotenv import load_dotenv
load_dotenv()

# Pre-test Ollama model availability check
import sys
from app.models.ollama_client import OllamaClient
from app.api import security

@pytest.fixture(scope="session", autouse=True)
def ensure_ollama_models():
    """Fail fast if no models are available in Ollama."""
    async def check_models():
        client = OllamaClient(base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        await client.initialize()
        models = await client.list_models(force_refresh=True)
        if not models:
            pytest.exit("No models available in Ollama. Aborting tests.", returncode=1)
    asyncio.get_event_loop().run_until_complete(check_models())


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """FastAPI test client with lifespan support (sync)"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client():
    """Async FastAPI test client with lifespan support (async)"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def mock_model_manager():
    """Mock model manager for testing"""
    # This would be a mock implementation for testing
    # without requiring actual Ollama connection
    class MockModelManager:
        async def initialize(self):
            pass
        
        async def generate(self, model_name, prompt, **kwargs):
            from app.models.manager import ModelResult
            return ModelResult(
                success=True,
                text="Mock response from model",
                cost=0.001,
                execution_time=0.5,
                model_used=model_name
            )
        
        def is_healthy(self):
            return True
        
        async def get_metrics(self):
            return {"total_requests": 0, "local_requests": 0}
    
    return MockModelManager()


@pytest.fixture
async def mock_cache_manager():
    """Mock cache manager for testing"""
    class MockCacheManager:
        def __init__(self):
            self._cache = {}
        
        async def initialize(self):
            pass
        
        async def get(self, key, default=None):
            return self._cache.get(key, default)
        
        async def set(self, key, value, ttl=None):
            self._cache[key] = value
            return True
        
        async def health_check(self):
            return True
        
        async def get_remaining_budget(self, user_id):
            return 100.0
        
        async def check_rate_limit(self, user_id, limit):
            return True, 1
        
        async def get_metrics(self):
            return {"cache_hits": 0, "cache_misses": 0}
    
    return MockCacheManager()


@pytest.fixture(autouse=True)
def override_get_current_user(monkeypatch):
    async def dummy_user(*args, **kwargs):
        return {"user_id": "test_user", "tier": "free"}
    monkeypatch.setattr(security, "get_current_user", dummy_user)


@pytest.fixture
def sample_graph_state():
    """Create a sample GraphState for testing."""
    return GraphState(
        original_query="What is artificial intelligence?",
        processed_query="What is artificial intelligence?",  # Add this line
        user_id="test_user_123",
        session_id="test_session_456",
        quality_requirement="balanced",  # Use string instead of QualityLevel enum
        cost_budget_remaining=0.5,
        max_cost=0.5,  # Add max_cost field
        max_execution_time=30.0
    )
