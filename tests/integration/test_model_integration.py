"""
End-to-end integration test for model management system.
Tests real Ollama integration with fallback handling.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import httpx

from app.models.ollama_client import OllamaClient, ModelResult, ModelStatus, OllamaException
from app.models.manager import ModelManager, TaskType, QualityLevel
from app.core.logging import setup_logging, get_correlation_id, set_correlation_id


# Setup logging for tests
setup_logging(log_level="DEBUG", log_format="text", enable_file_logging=False)


@pytest.fixture
async def mock_ollama_client():
    """Create a mock Ollama client for testing."""
    client = AsyncMock(spec=OllamaClient)
    
    # Mock successful health check
    client.health_check.return_value = True
    
    # Mock model list
    client.list_models.return_value = [
        {"name": "phi:mini", "size": "1.2GB"},
        {"name": "llama2:7b", "size": "3.8GB"},
        {"name": "mistral:7b", "size": "4.1GB"}
    ]
    
    # Mock model status checks
    client.check_model_status.return_value = ModelStatus.READY
    
    # Mock successful generation
    client.generate.return_value = ModelResult(
        success=True,
        text="This is a test response from the model.",
        execution_time=1.5,
        model_used="llama2:7b",
        tokens_generated=10,
        tokens_per_second=6.7,
        metadata={"total_duration": 1.5}
    )
    
    return client


@pytest.fixture
async def model_manager(mock_ollama_client):
    """Create a ModelManager with mocked Ollama client."""
    manager = ModelManager()
    manager.ollama_client = mock_ollama_client
    await manager.initialize()
    return manager


class TestOllamaClient:
    """Test OllamaClient functionality."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization and configuration."""
        client = OllamaClient(
            base_url="http://localhost:11434",
            timeout=30.0,
            max_retries=2
        )
        
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30.0
        assert client.max_retries == 2
        assert client._client is None
        
        await client.initialize()
        assert client._client is not None
        
        await client.close()
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_ollama_client):
        """Test successful health check."""
        # Test with real client that should fail gracefully
        client = OllamaClient()
        
        # Mock the HTTP response
        with patch.object(client, '_client') as mock_http_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_http_client.get.return_value = mock_response
            
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling."""
        client = OllamaClient()
        await client.initialize()
        
        # Mock HTTP client to raise exception
        with patch.object(client._client, 'get', side_effect=httpx.ConnectError("Connection failed")):
            result = await client.health_check()
            assert result is False
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_model_list_caching(self):
        """Test model list caching behavior."""
        client = OllamaClient()
        
        # Mock successful response
        mock_models = [{"name": "test:model", "size": "1GB"}]
        
        with patch.object(client, '_make_request', return_value={"models": mock_models}) as mock_request:
            # First call should make request
            result1 = await client.list_models()
            assert result1 == mock_models
            assert mock_request.call_count == 1
            
            # Second call should use cache
            result2 = await client.list_models()
            assert result2 == mock_models
            assert mock_request.call_count == 1  # Should not increase
            
            # Force refresh should make new request
            result3 = await client.list_models(force_refresh=True)
            assert result3 == mock_models
            assert mock_request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generation_with_retry(self):
        """Test generation with retry logic."""
        client = OllamaClient(max_retries=2, retry_delay=0.1)
        
        # Mock request that fails twice then succeeds
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("Temporary failure")
            return {
                "response": "Success after retries",
                "total_duration": 1000000000,  # 1 second in nanoseconds
                "eval_count": 5
            }
        
        with patch.object(client, '_make_request', side_effect=mock_request):
            result = await client.generate("test:model", "test prompt")
            
            assert result.success is True
            assert result.text == "Success after retries"
            assert call_count == 3  # Initial + 2 retries


class TestModelManager:
    """Test ModelManager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, mock_ollama_client):
        """Test ModelManager initialization."""
        manager = ModelManager()
        manager.ollama_client = mock_ollama_client
        
        await manager.initialize()
        
        # Verify initialization calls
        mock_ollama_client.initialize.assert_called_once()
        mock_ollama_client.health_check.assert_called_once()
        mock_ollama_client.list_models.assert_called_once()
        
        # Verify models were discovered
        assert len(manager.models) > 0
        assert "phi:mini" in manager.models
        assert "llama2:7b" in manager.models
    
    @pytest.mark.asyncio
    async def test_model_selection_by_task(self, model_manager):
        """Test model selection based on task type."""
        # Test different task types
        simple_model = model_manager.select_optimal_model(TaskType.SIMPLE_CLASSIFICATION)
        assert simple_model.startswith("phi")
        
        qa_model = model_manager.select_optimal_model(TaskType.QA_AND_SUMMARY)
        assert qa_model.startswith("llama2")
        
        reasoning_model = model_manager.select_optimal_model(TaskType.ANALYTICAL_REASONING)
        assert reasoning_model.startswith("mistral")
    
    @pytest.mark.asyncio
    async def test_model_selection_by_quality(self, model_manager):
        """Test model selection based on quality requirements."""
        # Minimal quality should prefer faster models
        minimal_model = model_manager.select_optimal_model(
            TaskType.QA_AND_SUMMARY,
            QualityLevel.MINIMAL
        )
        assert minimal_model.startswith("phi")
        
        # Premium quality should prefer better models
        premium_model = model_manager.select_optimal_model(
            TaskType.SIMPLE_CLASSIFICATION,
            QualityLevel.PREMIUM
        )
        assert premium_model.startswith("llama2")
    
    @pytest.mark.asyncio
    async def test_text_generation_success(self, model_manager):
        """Test successful text generation."""
        set_correlation_id("test-correlation-123")
        
        result = await model_manager.generate(
            model_name="llama2:7b",
            prompt="What is artificial intelligence?",
            max_tokens=100,
            temperature=0.7
        )
        
        assert result.success is True
        assert result.text == "This is a test response from the model."
        assert result.model_used == "llama2:7b"
        assert result.execution_time > 0
        assert result.cost == 0.0  # Local models are free
        
        # Verify model stats were updated
        model_info = model_manager.models["llama2:7b"]
        assert model_info.total_requests == 1
        assert model_info.avg_response_time > 0
    
    @pytest.mark.asyncio
    async def test_text_generation_with_fallback(self, model_manager):
        """Test text generation with fallback handling."""
        # Mock the primary model to fail
        model_manager.ollama_client.generate.side_effect = [
            OllamaException("Primary model failed"),
            ModelResult(
                success=True,
                text="Fallback response",
                execution_time=2.0,
                model_used="phi:mini"
            )
        ]
        
        result = await model_manager.generate(
            model_name="nonexistent:model",
            prompt="Test prompt",
            fallback=True
        )
        
        # Should succeed with fallback
        assert result.success is True
        assert result.text == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_model_preloading(self, model_manager):
        """Test concurrent model preloading."""
        models_to_preload = ["phi:mini", "llama2:7b"]
        
        results = await model_manager.preload_models(models_to_preload)
        
        # All models should load successfully (mocked)
        assert all(results.values())
        assert len(results) == 2
        
        # Models should be marked as loaded
        for model_name in models_to_preload:
            assert model_name in model_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_model_stats_collection(self, model_manager):
        """Test model statistics collection."""
        # Generate some requests to collect stats
        await model_manager.generate("llama2:7b", "Test prompt 1")
        await model_manager.generate("phi:mini", "Test prompt 2")
        
        stats = model_manager.get_model_stats()
        
        assert stats["total_models"] >= 2
        assert stats["loaded_models"] >= 0
        assert "model_details" in stats
        assert "performance_summary" in stats
        
        # Check model details
        if "llama2:7b" in stats["model_details"]:
            model_detail = stats["model_details"]["llama2:7b"]
            assert "total_requests" in model_detail
            assert "avg_response_time" in model_detail
            assert "success_rate" in model_detail
    
    @pytest.mark.asyncio
    async def test_model_recommendations(self, model_manager):
        """Test model optimization recommendations."""
        # Simulate some model usage with poor performance
        model_manager.models["llama2:7b"].avg_response_time = 15.0  # Slow
        model_manager.models["llama2:7b"].success_rate = 0.7  # Low success rate
        model_manager.models["llama2:7b"].total_requests = 10  # Enough data
        
        recommendations = model_manager.get_model_recommendations()
        
        assert "performance_optimizations" in recommendations
        assert "reliability_improvements" in recommendations
        assert len(recommendations["performance_optimizations"]) > 0
        assert len(recommendations["reliability_improvements"]) > 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(self, model_manager):
        """Test complete conversation flow from selection to generation."""
        set_correlation_id("e2e-test-456")
        
        # Step 1: Select model for conversation task
        selected_model = model_manager.select_optimal_model(
            TaskType.CONVERSATION,
            QualityLevel.BALANCED
        )
        
        assert selected_model is not None
        
        # Step 2: Generate response
        result = await model_manager.generate(
            model_name=selected_model,
            prompt="Hello, how are you today?",
            max_tokens=50,
            temperature=0.8
        )
        
        # Step 3: Verify successful generation
        assert result.success is True
        assert len(result.text) > 0
        assert result.cost == 0.0
        
        # Step 4: Check stats were updated
        stats = model_manager.get_model_stats()
        assert stats["performance_summary"]["total_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_generations(self, model_manager):
        """Test handling multiple concurrent generation requests."""
        set_correlation_id("concurrent-test-789")
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = model_manager.generate(
                model_name="llama2:7b",
                prompt=f"Test prompt {i}",
                max_tokens=20
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result.success for result in results)
        assert len(results) == 5
        
        # Model stats should reflect all requests
        model_info = model_manager.models["llama2:7b"]
        assert model_info.total_requests >= 5
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, model_manager):
        """Test graceful degradation when models fail."""
        set_correlation_id("degradation-test-101")
        
        # Mock Ollama client to simulate failures
        failure_count = 0
        original_generate = model_manager.ollama_client.generate
        
        async def failing_generate(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # First two calls fail
                raise OllamaException("Simulated model failure")
            return await original_generate(*args, **kwargs)
        
        model_manager.ollama_client.generate = failing_generate
        
        # Request should still succeed via fallback
        result = await model_manager.generate(
            model_name="mistral:7b",  # This will fail
            prompt="Test graceful degradation",
            fallback=True
        )
        
        # Should succeed with fallback model
        assert result.success is True
        assert result.text is not None
        
        # Verify fallback was used (different model)
        assert result.model_used != "mistral:7b"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_ollama_service_unavailable(self):
        """Test handling when Ollama service is unavailable."""
        # Create client with invalid URL
        client = OllamaClient(base_url="http://invalid-host:11434")
        
        # Health check should fail gracefully
        is_healthy = await client.health_check()
        assert is_healthy is False
        
        # Generation should fail gracefully
        result = await client.generate("test:model", "test prompt")
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_invalid_model_name(self, model_manager):
        """Test handling of invalid model names."""
        result = await model_manager.generate(
            model_name="nonexistent:model",
            prompt="Test prompt",
            fallback=True
        )
        
        # Should succeed via fallback
        assert result.success is True
        # Fallback model should be used
        assert result.model_used != "nonexistent:model"
    
    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self, model_manager):
        """Test handling of empty or invalid prompts."""
        # Empty prompt
        result = await model_manager.generate(
            model_name="llama2:7b",
            prompt="",
            fallback=False
        )
        
        # Should handle gracefully (Ollama might accept empty prompts)
        assert isinstance(result, ModelResult)
    
    @pytest.mark.asyncio
    async def test_manager_initialization_failure(self):
        """Test ModelManager initialization failure handling."""
        manager = ModelManager(ollama_host="http://invalid-host:11434")
        
        # Mock the ollama client to fail health check
        with patch.object(manager.ollama_client, 'health_check', return_value=False):
            with pytest.raises(OllamaException):
                await manager.initialize()
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loading(self, model_manager):
        """Test concurrent loading of the same model."""
        # Clear loaded models to force loading
        model_manager.loaded_models.clear()
        
        # Start multiple concurrent loads of the same model
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(
                model_manager._ensure_model_loaded("llama2:7b")
            )
            tasks.append(task)
        
        # All should complete successfully without conflicts
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Model should be loaded only once
        assert "llama2:7b" in model_manager.loaded_models


class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_response_time_tracking(self, model_manager):
        """Test response time tracking and averaging."""
        model_name = "llama2:7b"
        
        # Simulate multiple requests with different response times
        model_manager.ollama_client.generate.side_effect = [
            ModelResult(success=True, text="Response 1", execution_time=1.0, model_used=model_name),
            ModelResult(success=True, text="Response 2", execution_time=2.0, model_used=model_name),
            ModelResult(success=True, text="Response 3", execution_time=3.0, model_used=model_name)
        ]
        
        # Execute requests
        for i in range(3):
            await model_manager.generate(model_name, f"Prompt {i}")
        
        # Check average response time
        model_info = model_manager.models[model_name]
        assert model_info.total_requests == 3
        assert model_info.avg_response_time > 0
        
        # Average should be influenced by all requests
        assert 1.0 <= model_info.avg_response_time <= 3.0
    
    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, model_manager):
        """Test success rate calculation with mixed results."""
        model_name = "llama2:7b"
        
        # Simulate mixed success/failure results
        model_manager.ollama_client.generate.side_effect = [
            ModelResult(success=True, text="Success", execution_time=1.0, model_used=model_name),
            ModelResult(success=False, error="Failed", execution_time=0.5, model_used=model_name),
            ModelResult(success=True, text="Success", execution_time=1.2, model_used=model_name),
            ModelResult(success=True, text="Success", execution_time=0.8, model_used=model_name)
        ]
        
        # Execute requests
        for i in range(4):
            try:
                await model_manager.generate(model_name, f"Prompt {i}", fallback=False)
            except:
                pass  # Ignore failures for this test
        
        # Check success rate (3 out of 4 = 75%)
        model_info = model_manager.models[model_name]
        assert 0.7 <= model_info.success_rate <= 0.8  # Should be around 75%
    
    @pytest.mark.asyncio
    async def test_performance_score_calculation(self, model_manager):
        """Test overall performance score calculation."""
        model_name = "llama2:7b"
        model_info = model_manager.models[model_name]
        
        # Set specific metrics
        model_info.avg_response_time = 2.0  # Reasonable speed
        model_info.success_rate = 0.95      # High reliability
        model_info.confidence_scores = [0.8, 0.85, 0.9]  # Good confidence
        
        performance_score = model_info.performance_score
        
        # Score should be between 0 and 1
        assert 0.0 <= performance_score <= 1.0
        
        # With good metrics, score should be relatively high
        assert performance_score > 0.6


@pytest.mark.integration
class TestRealOllamaIntegration:
    """
    Integration tests that require a real Ollama instance.
    These are skipped by default and can be run when Ollama is available.
    """
    
    @pytest.mark.skip(reason="Requires running Ollama instance")
    @pytest.mark.asyncio
    async def test_real_ollama_health_check(self):
        """Test against real Ollama instance."""
        client = OllamaClient()
        await client.initialize()
        
        try:
            is_healthy = await client.health_check()
            # If Ollama is running, this should pass
            assert is_healthy is True
            
            # Test model listing
            models = await client.list_models()
            assert isinstance(models, list)
            
        finally:
            await client.close()
    
    @pytest.mark.skip(reason="Requires running Ollama instance with models")
    @pytest.mark.asyncio
    async def test_real_text_generation(self):
        """Test real text generation."""
        manager = ModelManager()
        
        try:
            await manager.initialize()
            
            # Try to generate with any available model
            stats = manager.get_model_stats()
            if stats["total_models"] > 0:
                available_models = list(manager.models.keys())
                test_model = available_models[0]
                
                result = await manager.generate(
                    model_name=test_model,
                    prompt="What is 2 + 2?",
                    max_tokens=10
                )
                
                assert result.success is True
                assert len(result.text) > 0
                assert result.execution_time > 0
        
        finally:
            await manager.cleanup()


if __name__ == "__main__":
    # Run specific test for development
    import asyncio
    
    async def run_basic_test():
        """Run a basic test for development."""
        print("Running basic model integration test...")
        
        # Setup correlation ID
        set_correlation_id("dev-test-001")
        
        # Test Ollama client
        client = OllamaClient()
        
        try:
            await client.initialize()
            print(f"✓ OllamaClient initialized")
            
            # Test health check
            is_healthy = await client.health_check()
            print(f"✓ Health check: {'Healthy' if is_healthy else 'Unhealthy'}")
            
            if is_healthy:
                # Test model listing
                models = await client.list_models()
                print(f"✓ Found {len(models)} models")
                
                # Test ModelManager
                manager = ModelManager()
                manager.ollama_client = client
                await manager.initialize()
                print(f"✓ ModelManager initialized with {len(manager.models)} models")
                
                # Test model selection
                selected_model = manager.select_optimal_model(TaskType.CONVERSATION)
                print(f"✓ Selected model for conversation: {selected_model}")
                
                print("\n🎉 All basic tests passed!")
            else:
                print("⚠️  Ollama not available - some tests skipped")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        finally:
            await client.close()
    
    # Run the test
    asyncio.run(run_basic_test())
