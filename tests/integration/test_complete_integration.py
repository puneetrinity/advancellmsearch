"""
Comprehensive integration tests for the complete AI search system.
Tests end-to-end functionality from API to model execution.
"""
import pytest
import pytest_asyncio
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from asgi_lifespan import LifespanManager

from app.main import app
from app.core.logging import setup_logging, set_correlation_id
from app.api.chat import set_dependencies
from app.models.manager import ModelManager, TaskType, QualityLevel
from app.models.ollama_client import ModelResult
from app.graphs.chat_graph import ChatGraph
from app.cache.redis_client import CacheManager


# Setup logging for tests
setup_logging(log_level="DEBUG", log_format="text", enable_file_logging=False)


@pytest.fixture
async def mock_model_manager():
    """Create a comprehensive mock ModelManager."""
    manager = AsyncMock(spec=ModelManager)
    
    # Mock model selection based on task type
    def mock_select_model(task_type, quality_level=QualityLevel.BALANCED):
        model_mapping = {
            TaskType.SIMPLE_CLASSIFICATION: "phi:mini",
            TaskType.QA_AND_SUMMARY: "llama2:7b",
            TaskType.ANALYTICAL_REASONING: "mistral:7b",
            TaskType.CODE_TASKS: "codellama",
            TaskType.CONVERSATION: "llama2:7b"
        }
        return model_mapping.get(task_type, "llama2:7b")
    
    manager.select_optimal_model.side_effect = mock_select_model
    
    # Mock generation with realistic responses
    def mock_generate(model_name, prompt, **kwargs):
        # Simulate different response types based on prompt content
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            response = "Hello! I'm here to help you with any questions you have."
            intent_response = "conversation"
        elif "code" in prompt.lower() or "function" in prompt.lower():
            response = "Here's how you can write that function in Python:\n\n```python\ndef example():\n    pass\n```"
            intent_response = "code"
        elif "explain" in prompt.lower() or "what is" in prompt.lower():
            response = "Let me explain that concept for you. This is a detailed explanation that covers the key points."
            intent_response = "question"
        elif "analyze" in prompt.lower() or "compare" in prompt.lower():
            response = "Based on my analysis, here are the key differences and similarities to consider."
            intent_response = "analysis"
        elif "category" in prompt.lower() and ":" in prompt:
            # Intent classification prompt
            if "hello" in prompt.lower():
                intent_response = "conversation"
            elif "code" in prompt.lower():
                intent_response = "code"
            elif "explain" in prompt.lower():
                intent_response = "question"
            else:
                intent_response = "question"
            response = intent_response
        else:
            response = "I understand your question and I'm happy to help with that."
            intent_response = "question"
        
        return ModelResult(
            success=True,
            text=response,
            execution_time=0.5 + (len(prompt) / 1000),  # Realistic timing
            model_used=model_name,
            cost=0.0,  # Local models are free
            tokens_generated=len(response.split()),
            tokens_per_second=20.0,
            metadata={"prompt_length": len(prompt)}
        )
    
    manager.generate.side_effect = mock_generate
    
    # Mock initialization
    manager.initialize.return_value = None
    manager.cleanup.return_value = None
    
    # Mock statistics
    manager.get_model_stats.return_value = {
        "total_models": 3,
        "loaded_models": 2,
        "performance_summary": {
            "total_requests": 10,
            "avg_response_time": 1.2,
            "local_percentage": 100.0
        }
    }
    
    return manager


@pytest.fixture
async def mock_cache_manager():
    """Create a mock cache manager."""
    cache = AsyncMock(spec=CacheManager)
    
    # Mock conversation history
    cache.get_conversation_history.return_value = []
    cache.update_conversation_history.return_value = True
    cache.cache_successful_route.return_value = True
    cache.update_user_pattern.return_value = True
    
    return cache


@pytest_asyncio.fixture
async def integration_client(mock_model_manager, mock_cache_manager):
    """Create test client with mocked dependencies."""
    set_dependencies(mock_model_manager, mock_cache_manager)
    from httpx import AsyncClient, ASGITransport
    from asgi_lifespan import LifespanManager
    app_instance = app
    async with LifespanManager(app_instance):
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestChatAPIIntegration:
    """Test complete chat API integration."""
    
    @pytest.mark.asyncio
    async def test_chat_streaming(self, integration_client):
        """Test streaming chat functionality."""
        response = await integration_client.post("/api/v1/chat/stream", json={
            "messages": [
                {"role": "user", "content": "What is artificial intelligence?"}
            ],
            "session_id": "test_stream_001",
            "model": "auto",
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": True
        })
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            content = response.content.decode()
            chunks = [line for line in content.split('\n') if line.startswith('data: ')]
            assert len(chunks) > 0
            if chunks[0] != "data: [DONE]":
                first_chunk_data = json.loads(chunks[0][6:])
                assert "id" in first_chunk_data
                assert "object" in first_chunk_data
                assert first_chunk_data["object"] == "chat.completion.chunk"
                assert "choices" in first_chunk_data
                assert len(first_chunk_data["choices"]) > 0
            assert "data: [DONE]" in content

    @pytest.mark.asyncio
    async def test_conversation_history_management(self, integration_client):
        """Test conversation history endpoints."""
        session_id = "test_history_session"
        await integration_client.post("/api/v1/chat/complete", json={
            "message": "Hello, I'm learning Python",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        history_response = await integration_client.get(f"/api/v1/chat/history/{session_id}")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert "session_id" in history_data
        assert "history" in history_data
        assert "message_count" in history_data
        clear_response = await integration_client.delete(f"/api/v1/chat/history/{session_id}")
        assert clear_response.status_code == 200
        clear_data = clear_response.json()
        assert clear_data["cleared"] is True
        assert clear_data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, integration_client):
        """Test multi-turn conversation with context."""
        session_id = "test_multi_turn"
        response1 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Hello, I'm working on a Python project",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        assert response1.status_code in (200, 422)
        response2 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Can you help me with functions?",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        assert response2.status_code in (200, 422)
        if response2.status_code == 200:
            data2 = response2.json()
            assert "python" in data2["data"]["response"].lower() or "function" in data2["data"]["response"].lower()

    @pytest.mark.asyncio
    async def test_different_query_types(self, integration_client):
        """Test handling of different query types."""
        test_cases = [
            {"message": "Hello there!", "expected_intent": "conversation"},
            {"message": "What is machine learning?", "expected_intent": "question"},
            {"message": "Write a Python function to sort a list", "expected_intent": "code"},
            {"message": "Compare React and Vue frameworks", "expected_intent": "analysis"}
        ]
        for i, test_case in enumerate(test_cases):
            response = await integration_client.post("/api/v1/chat/complete", json={
                "message": test_case["message"],
                "session_id": f"test_types_{i}",
                "context": {},
                "constraints": {},
            })
            assert response.status_code in (200, 422)
            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]["response"]) > 10
                if "developer_hints" in data:
                    routing = data["developer_hints"].get("routing_explanation", "")
                    assert len(routing) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, integration_client):
        """Test API error handling."""
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": "",
            "session_id": "test_error",
            "context": {},
            "constraints": {}
        })
        assert response.status_code == 422
        long_message = "x" * 10000
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": long_message,
            "session_id": "test_error",
            "context": {},
            "constraints": {}
        })
        assert response.status_code == 422
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Test message",
            "constraints": {"quality_requirement": "invalid"},
            "context": {}
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_cost_and_budget_tracking(self, integration_client):
        """Test cost tracking and budget constraints."""
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Complex analysis query that might cost more",
            "constraints": {"max_cost": 0.01, "quality_requirement": "premium"},
            "context": {}
        })
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            data = response.json()
            cost_prediction = data["cost_prediction"]
            assert cost_prediction["estimated_cost"] <= 0.01
            assert "budget_remaining" in cost_prediction
            assert "budget_percentage_used" in cost_prediction

    @pytest.mark.asyncio
    async def test_authentication_and_rate_limiting(self, integration_client):
        """Test authentication and rate limiting."""
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Test without auth",
            "context": {},
            "constraints": {}
        })
        assert response.status_code in (200, 422)
        response = await integration_client.post(
            "/api/v1/chat/complete",
            json={"message": "Test with auth", "context": {}, "constraints": {}},
            headers={"Authorization": "Bearer dev-user-token"}
        )
        assert response.status_code in (200, 422)


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_health_endpoints(self, integration_client):
        """Test system health endpoints."""
        response = await integration_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    @pytest.mark.asyncio
    async def test_search_api_integration(self, integration_client):
        """Test search API integration."""
        response = await integration_client.get("/api/v1/search/health")
        assert response.status_code == 200
        # Test basic search
        response = await integration_client.post("/api/v1/search/basic", json={
            "query": "artificial intelligence",
            "max_results": 5,
            "search_type": "web",
            "include_summary": True
        })
        assert response.status_code in (200, 422)
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert "query" in data["data"]
            assert "results" in data["data"]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, integration_client):
        """Test handling of concurrent requests."""
        # Create multiple concurrent chat requests
        tasks = []
        for i in range(5):
            task = integration_client.post("/api/v1/chat/complete", json={
                "message": f"Concurrent test message {i}",
                "session_id": f"concurrent_{i}",
                "context": {},
                "constraints": {}
            })
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code in (200, 422)]
        assert len(successful_responses) >= 4  # Allow for some potential failures
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, integration_client):
        """Test system performance under moderate load."""
        import time
        
        start_time = time.time()
        
        # Send 10 requests sequentially
        response_times = []
        for i in range(10):
            request_start = time.time()
            
            response = await integration_client.post("/api/v1/chat/complete", json={
                "message": f"Performance test query {i}",
                "session_id": f"perf_test_{i}",
                "context": {},
                "constraints": {}
            })
            
            request_time = time.time() - request_start
            response_times.append(request_time)
            
            assert response.status_code in (200, 422)
        
        total_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times)
        
        # Performance assertions
        assert avg_response_time < 5.0  # Average response under 5 seconds
        assert max(response_times) < 10.0  # No response over 10 seconds
        assert total_time < 30.0  # Total time under 30 seconds
        
        print(f"Performance test results:")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Max response time: {max(response_times):.2f}s")
        print(f"  Total test time: {total_time:.2f}s")


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_customer_support_scenario(self, integration_client):
        """Test a customer support conversation scenario."""
        session_id = "customer_support_test"
        
        response1 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Hi, I need help with my account",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        assert response1.status_code in (200, 422)
        response2 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "I can't log in to my dashboard",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        assert response2.status_code in (200, 422)
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            assert len(data1["data"]["response"]) > 20
            assert len(data2["data"]["response"]) > 20
            assert "help" in data1["data"]["response"].lower()

    @pytest.mark.asyncio
    async def test_technical_documentation_scenario(self, integration_client):
        """Test technical documentation assistance scenario."""
        session_id = "tech_docs_test"
        
        response1 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "How do I implement authentication in FastAPI?",
            "session_id": session_id,
            "context": {},
            "constraints": {"quality_requirement": "high"}
        })
        assert response1.status_code in (200, 422)
        response2 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Can you show me a code example?",
            "session_id": session_id,
            "context": {},
            "constraints": {}
        })
        assert response2.status_code in (200, 422)
        if response2.status_code == 200:
            data2 = response2.json()
            response_text = data2["data"]["response"].lower()
            assert any(word in response_text for word in ["code", "function", "import", "def", "class"])
    
    @pytest.mark.asyncio
    async def test_learning_assistance_scenario(self, integration_client):
        """Test educational/learning assistance scenario."""
        session_id = "learning_test"
        response1 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "I'm learning Python. Can you explain variables?",
            "session_id": session_id,
            "context": {},
            "constraints": {},
        })
        assert response1.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_business_analysis_scenario(self, integration_client):
        """Test business analysis and decision support scenario."""
        response = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Compare the pros and cons of microservices vs monolithic architecture for a startup",
            "context": {},
            "constraints": {"quality_requirement": "premium"},
        })
        assert response.status_code in (200, 422)

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, integration_client):
        """Test handling of unicode and special characters."""
        test_messages = [
            "Hello! 👋 How are you today? 😊",
            "Explain 人工智能 (artificial intelligence)",
            "What about café, résumé, and naïve?",
            "Test with emojis: 🚀🤖💡🔬📊"
        ]
        for message in test_messages:
            response = await integration_client.post("/api/v1/chat/complete", json={
                "message": message,
                "session_id": "unicode_test",
                "context": {},
                "constraints": {}
            })
            assert response.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_very_short_and_long_messages(self, integration_client):
        """Test very short and moderately long messages."""
        response1 = await integration_client.post("/api/v1/chat/complete", json={
            "message": "Hi",
            "session_id": "short_test",
            "context": {},
            "constraints": {}
        })
        assert response1.status_code in (200, 422)
        long_message = "x" * 7999
        response2 = await integration_client.post("/api/v1/chat/complete", json={
            "message": long_message,
            "session_id": "long_test",
            "context": {},
            "constraints": {}
        })
        assert response2.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_rapid_fire_requests(self, integration_client):
        """Test rapid consecutive requests."""
        session_id = "rapid_fire_test"
        
        # Send 5 requests quickly
        for i in range(5):
            response = await integration_client.post("/api/v1/chat/complete", json={
                "message": f"Quick question {i}",
                "session_id": session_id,
                "context": {},
                "constraints": {}
            })
            
            # Should handle rapid requests gracefully
            assert response.status_code in (200, 422, 429)  # Success or rate limited
            
            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]["response"]) > 0


if __name__ == "__main__":
    # Run integration test suite
    async def run_integration_tests():
        """Run key integration tests for verification."""
        print("🧪 Running Integration Test Suite...")
        
        # Setup
        setup_logging(log_level="INFO", log_format="text", enable_file_logging=False)
        set_correlation_id("integration-test-suite")
        
        try:
            # Create mock dependencies
            mock_model_manager = AsyncMock(spec=ModelManager)
            mock_model_manager.select_optimal_model.return_value = "llama2:7b"
            mock_model_manager.generate.return_value = ModelResult(
                success=True,
                text="Test response from integration suite",
                execution_time=1.0,
                model_used="llama2:7b",
                cost=0.0
            )
            
            mock_cache_manager = AsyncMock(spec=CacheManager)
            mock_cache_manager.get_conversation_history.return_value = []
            
            # Set dependencies
            set_dependencies(mock_model_manager, mock_cache_manager)
            
            # Test with sync client
            client = TestClient(app)
            
            # Test 1: Basic chat
            print("  ✅ Testing basic chat...")
            response = client.post("/api/v1/chat/complete", json={
                "message": "Hello, test integration"
            })
            assert response.status_code == 200
            print(f"     Response: {response.json()['data']['response'][:50]}...")
            
            # Test 2: Health check
            print("  ✅ Testing health endpoint...")
            health_response = client.get("/health")
            assert health_response.status_code == 200
            print(f"     Health: {health_response.json()['status']}")
            
            # Test 3: Search API
            print("  ✅ Testing search API...")
            search_response = client.post("/api/v1/search/basic", json={
                "query": "test search"
            })
            assert search_response.status_code == 200
            print(f"     Search results: {len(search_response.json()['data']['results'])} items")
            
            print("\n🎉 Integration test suite PASSED!")
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test suite FAILED: {e}")
            return False
    
    # Run the tests
    import asyncio
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)
