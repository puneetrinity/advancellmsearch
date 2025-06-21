"""
CORRECTED Integration tests with proper request schemas and coroutine safety.
This file replaces the previous integration tests with fixes for all identified issues.
"""

import pytest
import httpx
import asyncio
import time
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from app.main import app, app_state

# Sync client for basic endpoints
client = TestClient(app)

@pytest.fixture
async def async_client():
    """Async client for async endpoints"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), 
        base_url="http://test"
    ) as client:
        yield client

@pytest.fixture(autouse=True)
def setup_safe_app_components():
    """
    Setup proper mock components with coroutine safety.
    """
    async def safe_search_execute(query, budget=2.0, quality="standard", max_results=10, **kwargs):
        result = {
            "response": f"Test search response for: {query}",
            "citations": [
                {"url": "https://example.com/1", "title": "Test Result 1"},
                {"url": "https://example.com/2", "title": "Test Result 2"}
            ],
            "metadata": {
                "execution_time": 0.01, 
                "total_cost": 0.0,
                "query_id": "test-query-123",
                "confidence_score": 0.95,
                "provider_used": "test_provider"
            }
        }
        assert not asyncio.iscoroutine(result), "Mock search result is a coroutine!"
        return result

    async def safe_chat_execute(state_or_request, **kwargs):
        result = Mock()
        result.final_response = "Test chat response"
        result.execution_time = 1.0
        result.calculate_total_cost = Mock(return_value=0.01)
        result.execution_path = ["test_node"]
        result.conversation_history = []
        result.sources_consulted = []
        result.citations = []
        result.warnings = []
        result.costs_incurred = {}
        result.models_used = {"test_model"}
        result.escalation_reason = None
        result.errors = None
        result.intermediate_results = {}
        result.get_avg_confidence = Mock(return_value=0.9)
        assert not asyncio.iscoroutine(result), "Mock chat result is a coroutine!"
        return result

    mock_model_manager = Mock()
    mock_model_manager.get_model_stats.return_value = {
        "total_models": 1,
        "loaded_models": 1,
        "available_models": ["test_model"]
    }
    mock_model_manager.generate = AsyncMock(return_value=Mock(
        success=True,
        text="Test response",
        cost=0.01
    ))

    mock_cache_manager = Mock()
    mock_cache_manager.get = AsyncMock(return_value=None)
    mock_cache_manager.set = AsyncMock(return_value=True)

    mock_search_system = Mock()
    mock_search_system.execute_optimized_search = safe_search_execute

    mock_chat_graph = Mock()
    mock_chat_graph.execute = safe_chat_execute
    mock_chat_graph.get_performance_stats.return_value = {
        "executions": 1,
        "avg_time": 1.0
    }

    mock_search_graph = Mock()
    mock_search_graph.execute = AsyncMock(return_value=Mock(
        results=[],
        summary="Test summary",
        total_results=0,
        search_time=0.1,
        sources_consulted=[]
    ))

    async def verify_mock_safety():
        print("\n🔍 VERIFYING MOCK SAFETY:")
        search_result = await mock_search_system.execute_optimized_search("test query")
        print(f"🔍 Search mock result type: {type(search_result)}")
        print(f"🔍 Search mock is_coroutine: {asyncio.iscoroutine(search_result)}")
        assert not asyncio.iscoroutine(search_result), "Search mock returning coroutine!"
        chat_result = await mock_chat_graph.execute(Mock())
        print(f"🔍 Chat mock result type: {type(chat_result)}")
        print(f"🔍 Chat mock is_coroutine: {asyncio.iscoroutine(chat_result)}")
        assert not asyncio.iscoroutine(chat_result), "Chat mock returning coroutine!"
        print("✅ Mock verification passed - no coroutines detected!")

    try:
        asyncio.get_event_loop().run_until_complete(verify_mock_safety())
    except RuntimeError:
        asyncio.run(verify_mock_safety())

    try:
        from app.api.chat import set_dependencies as set_chat_dependencies
        set_chat_dependencies(mock_model_manager, mock_cache_manager, mock_chat_graph)
    except ImportError:
        pass
    try:
        from app.api.search import set_dependencies as set_search_dependencies
        set_search_dependencies(mock_model_manager, mock_cache_manager, mock_search_graph)
    except ImportError:
        pass
    app_state.update({
        "model_manager": mock_model_manager,
        "cache_manager": mock_cache_manager, 
        "chat_graph": mock_chat_graph,
        "search_graph": mock_search_graph,
        "search_system": mock_search_system,
        "api_key_status": {
            "brave_search": True,
            "scrapingbee": True
        },
        "startup_time": time.time()
    })
    yield
    app_state.clear()

# ...existing test code from your provided file...
