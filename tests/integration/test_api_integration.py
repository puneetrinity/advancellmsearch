# tests/integration/test_api_integration.py
"""
Integration tests for the full API
"""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, app_state

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_app_components():
    """Mock required components for testing - CORRECTED VERSION with debug verification"""
    import asyncio
    from unittest.mock import AsyncMock, Mock

    # Mock model manager with SYNC methods
    mock_model_manager = Mock()
    mock_model_manager.get_model_stats.return_value = {
        "total_models": 1,
        "loaded_models": 0,
        "model_details": {},
    }
    mock_model_manager.generate = AsyncMock(
        return_value=Mock(success=True, text="Test response", cost=0.01)
    )
    # Mock cache manager
    mock_cache_manager = Mock()
    mock_cache_manager.get = AsyncMock(return_value=None)
    mock_cache_manager.set = AsyncMock(return_value=True)
    # Mock chat graph with async execute that returns actual data
    mock_chat_graph = Mock()
    mock_chat_graph.execute = AsyncMock(
        return_value=Mock(
            final_response="Test chat response",
            execution_time=1.0,
            calculate_total_cost=Mock(return_value=0.01),
            execution_path=["test_node"],
            conversation_history=[],
            sources_consulted=[],
            citations=[],
            warnings=[],
            costs_incurred={},
            models_used=set(),
            escalation_reason=None,
            errors=None,
            intermediate_results={},
            get_avg_confidence=Mock(return_value=1.0),
        )
    )
    mock_chat_graph.get_performance_stats.return_value = {
        "executions": 1,
        "avg_time": 1.0,
    }
    # Mock search graph
    mock_search_graph = Mock()
    mock_search_graph.execute = AsyncMock(
        return_value=Mock(
            results=[],
            summary="Test summary",
            total_results=0,
            search_time=0.1,
            sources_consulted=[],
        )
    )
    # Mock search system
    mock_search_system = Mock()
    mock_search_system.execute_optimized_search = AsyncMock(
        return_value={
            "response": "Test search response",
            "citations": [],
            "metadata": {"execution_time": 0.01, "total_cost": 0.0},
        }
    )

    # 🔥 FIX: Properly configured search system mock

    async def mock_search_execute(
        query, budget=2.0, quality="standard", max_results=10, **kwargs
    ):
        # Return the actual dict, not a coroutine
        return {
            "response": f"Test search response for: {query}",
            "citations": [],
            "metadata": {
                "execution_time": 0.01,
                "total_cost": 0.0,
                "query_id": "test-query-123",
            },
        }

    mock_search_system.execute_optimized_search = mock_search_execute

    # 🔥 FIX: Properly configured chat graph mock

    async def mock_chat_execute(state_or_request, **kwargs):
        # Return actual result object, not coroutine
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
        result.models_used = set()
        result.escalation_reason = None
        result.errors = None
        result.intermediate_results = {}
        result.get_avg_confidence = Mock(return_value=1.0)
        return result

    mock_chat_graph.execute = mock_chat_execute

    # 🔥 VERIFICATION: Test mocks work correctly

    async def verify_mocks():
        print("\n🔍 VERIFYING MOCKS:")
        # Test search mock
        search_result = await mock_search_system.execute_optimized_search("test query")
        print(f"🔍 Search mock result type: {type(search_result)}")
        print(f"🔍 Search mock is_coroutine: "
            "{asyncio.iscoroutine(search_result)}")
        print(f"🔍 Search mock result: {search_result}")
        # Test chat mock
        chat_result = await mock_chat_graph.execute(Mock())
        print(f"🔍 Chat mock result type: {type(chat_result)}")
        print(f"🔍 Chat mock is_coroutine: {asyncio.iscoroutine(chat_result)}")
        # These should be False!
        assert not asyncio.iscoroutine(
            search_result
        ), "Search mock returning coroutine!"
        assert not asyncio.iscoroutine(chat_result), "Chat mock returning coroutine!"
        print("✅ Mock verification passed!")

    try:
        asyncio.get_event_loop().run_until_complete(verify_mocks())
    except RuntimeError:
        asyncio.run(verify_mocks())

    # Set FastAPI endpoint dependencies to use test mocks
    # from app.api.chat import set_dependencies as set_chat_dependencies
    # set_chat_dependencies(
        mock_model_manager,
        mock_cache_manager,
        mock_chat_graph
    
    # If search.set_dependencies exists, do the same for search
    # try:
    #     from app.api.search import set_dependencies as set_search_dependencies
    #     set_search_dependencies(
        mock_model_manager,
        mock_cache_manager,
        mock_search_graph
    )
    # except ImportError:
    #     pass

    # Update app_state
    app_state.update(
        {
            "model_manager": mock_model_manager,
            "cache_manager": mock_cache_manager,
            "chat_graph": mock_chat_graph,
            "search_graph": mock_search_graph,
            "search_system": mock_search_system,
            "api_key_status": {"brave_search": False, "scrapingbee": False},
            "startup_time": 1234567890.0,
        }
    )
    yield
    app_state.clear()


# Sync endpoints (health, root)
def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") in ("healthy", "degraded")


def test_root():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200


# Async endpoints (search, chat, metrics)
@pytest.mark.asyncio


async def test_search_basic():
    """Test basic search endpoint with correct wrapper and schema"""
    payload = {
        "query": "test search query",
        "max_results": 10,
        "search_type": "web",
        "include_summary": True,
        "budget": 2.0,
        "quality": "standard",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/search/basic", json=payload)
        if resp.status_code != 200:
            print("\n--- 422 Error Details ---\n", resp.text)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "metadata" in data


@pytest.mark.asyncio
async def test_chat_complete():
    """Test chat complete endpoint with correct wrapper and schema"""
    payload = {
        "message": "Hello, this is a test message",
        "session_id": "test-session-123",
        "user_context": {},
        "quality_requirement": "balanced",
        "max_cost": 0.10,
        "max_execution_time": 30.0,
        "force_local_only": False,
        "response_style": "balanced",
        "include_sources": True,
        "include_debug_info": False,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/chat/complete", json=payload)
        if resp.status_code != 200:
            print("\n--- 422 Error Details ---\n", resp.text)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ["success", "error"]
        assert "data" in data
        assert "metadata" in data
        assert "correlation_id" in data["metadata"]


@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "operational"


def test_search_test():
    payload = {"query": "test"}
    resp = client.post("/api/v1/search/test", json=payload)
    assert resp.status_code == 200
    assert "mock_results" in resp.json() and resp.json().get("status") == "success"
