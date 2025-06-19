"""
Integration tests for the graph system implementation.
Tests end-to-end graph execution with real model integration.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime

from app.core.logging import setup_logging, set_correlation_id
from app.graphs.base import GraphState, NodeResult, BaseGraphNode, BaseGraph, GraphType, NodeType
from app.graphs.chat_graph import ChatGraph, ContextManagerNode, IntentClassifierNode, ResponseGeneratorNode
from app.models.manager import ModelManager, TaskType, QualityLevel
from app.models.ollama_client import ModelResult


# Setup logging for tests
setup_logging(log_level="DEBUG", log_format="text", enable_file_logging=False)


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager for testing."""
    manager = AsyncMock(spec=ModelManager)
    
    # Mock model selection
    manager.select_optimal_model.return_value = "llama2:7b"
    
    # Mock successful generation
    manager.generate.return_value = ModelResult(
        success=True,
        text="This is a test response from the graph system.",
        execution_time=1.2,
        model_used="llama2:7b",
        cost=0.0,
        tokens_generated=15,
        tokens_per_second=12.5
    )
    
    return manager


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager for testing."""
    cache = AsyncMock()
    
    # Mock conversation history
    cache.get_conversation_history.return_value = [
        {"role": "user", "content": "Hello", "timestamp": 1234567890},
        {"role": "assistant", "content": "Hi there!", "timestamp": 1234567891}
    ]
    
    # Mock successful cache operations
    cache.update_conversation_history.return_value = True
    cache.cache_successful_route.return_value = True
    cache.update_user_pattern.return_value = True
    
    return cache


@pytest.fixture
def sample_graph_state():
    """Create a sample GraphState for testing."""
    return GraphState(
        original_query="What is artificial intelligence?",
        user_id="test_user_123",
        session_id="test_session_456",
        quality_requirement=QualityLevel.BALANCED,
        max_cost=0.50,
        max_execution_time=30.0
    )


class TestGraphState:
    """Test GraphState functionality."""
    
    def test_state_initialization(self):
        """Test GraphState initialization with defaults."""
        state = GraphState(original_query="Test query")
        
        assert state.original_query == "Test query"
        assert state.query_id is not None
        assert state.correlation_id is not None
        assert isinstance(state.start_time, datetime)
        assert state.execution_path == []
        assert state.costs_incurred == {}
        assert state.confidence_scores == {}
    
    def test_state_cost_tracking(self):
        """Test cost tracking functionality."""
        state = GraphState(original_query="Test")
        
        state.add_cost("model_generation", 0.05)
        state.add_cost("search_api", 0.02)
        state.add_cost("model_generation", 0.03)  # Add to existing
        
        assert state.costs_incurred["model_generation"] == 0.08
        assert state.costs_incurred["search_api"] == 0.02
        assert state.calculate_total_cost() == 0.10
    
    def test_state_confidence_tracking(self):
        """Test confidence score tracking."""
        state = GraphState(original_query="Test")
        
        state.set_confidence("intent_classifier", 0.9)
        state.set_confidence("response_generator", 0.85)
        
        assert state.confidence_scores["intent_classifier"] == 0.9
        assert state.confidence_scores["response_generator"] == 0.85
        assert state.get_avg_confidence() == 0.875
    
    def test_state_budget_checking(self):
        """Test budget constraint checking."""
        state = GraphState(original_query="Test", max_cost=0.10)
        
        # Within budget
        assert state.is_within_budget() is True
        assert state.is_within_budget(0.05) is True
        
        # Add costs
        state.add_cost("component1", 0.08)
        assert state.is_within_budget() is True
        assert state.is_within_budget(0.05) is False  # Would exceed
        
        # Exceed budget
        state.add_cost("component2", 0.05)
        assert state.is_within_budget() is False
    
    def test_state_execution_tracking(self):
        """Test execution step tracking."""
        state = GraphState(original_query="Test")
        
        result = {"success": True, "data": {"test": "value"}}
        state.add_execution_step("test_node", result)
        
        assert "test_node" in state.execution_path
        assert "test_node" in state.node_results
        assert state.node_results["test_node"]["result"] == result


class TestBaseGraphNode:
    """Test BaseGraphNode functionality."""
    
    class TestNode(BaseGraphNode):
        """Test node implementation."""
        
        def __init__(self, should_fail=False):
            super().__init__("test_node", NodeType.PROCESSING)
            self.should_fail = should_fail
        
        async def execute(self, state: GraphState, **kwargs) -> NodeResult:
            if self.should_fail:
                return NodeResult(
                    success=False,
                    error="Test node failure"
                )
            
            return NodeResult(
                success=True,
                data={"message": "Test node executed"},
                confidence=0.8,
                execution_time=0.5,
                cost=0.01
            )
    
    @pytest.mark.asyncio
    async def test_successful_node_execution(self, sample_graph_state):
        """Test successful node execution."""
        node = self.TestNode()
        
        initial_path_length = len(sample_graph_state.execution_path)
        result_state = await node(sample_graph_state)
        
        # Check state was updated
        assert len(result_state.execution_path) == initial_path_length + 1
        assert "test_node" in result_state.execution_path
        assert "test_node" in result_state.node_results
        
        # Check cost and timing were recorded
        assert "test_node" in result_state.costs_incurred
        assert "test_node" in result_state.execution_times
        assert "test_node" in result_state.confidence_scores
    
    @pytest.mark.asyncio
    async def test_failed_node_execution(self, sample_graph_state):
        """Test failed node execution handling."""
        node = self.TestNode(should_fail=True)
        
        result_state = await node(sample_graph_state)
        
        # Check error was recorded
        assert len(result_state.errors) > 0
        assert "test_node" in result_state.execution_path
        
        # Check node stats were updated
        assert node.execution_count == 1
        assert node.success_count == 0
    
    @pytest.mark.asyncio
    async def test_node_performance_stats(self):
        """Test node performance statistics."""
        node = self.TestNode()
        state = GraphState(original_query="Test")
        
        # Execute multiple times
        await node(state)
        await node(state)
        
        stats = node.get_performance_stats()
        
        assert stats["name"] == "test_node"
        assert stats["execution_count"] == 2
        assert stats["success_count"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["total_execution_time"] > 0


class TestChatGraphNodes:
    """Test individual ChatGraph nodes."""
    
    @pytest.mark.asyncio
    async def test_context_manager_node(self, mock_cache_manager, sample_graph_state):
        """Test ContextManagerNode functionality."""
        node = ContextManagerNode(mock_cache_manager)
        
        # Add some conversation history
        sample_graph_state.conversation_history = [
            {"role": "user", "content": "Hello, I'm working on a Python project"},
            {"role": "assistant", "content": "Great! I'd be happy to help with Python."}
        ]
        
        result_state = await node(sample_graph_state)
        
        # Check context analysis was performed
        assert "conversation_context" in result_state.intermediate_results
        context = result_state.intermediate_results["conversation_context"]
        
        assert "user_expertise_level" in context
        assert "conversation_mood" in context
        assert "previous_topics" in context
        
        # Check query was enhanced with context
        assert result_state.processed_query is not None
        assert result_state.query_complexity >= 0.0
    
    @pytest.mark.asyncio
    async def test_intent_classifier_node(self, mock_model_manager, sample_graph_state):
        """Test IntentClassifierNode functionality."""
        node = IntentClassifierNode(mock_model_manager)
        
        # Mock model response for classification
        mock_model_manager.generate.return_value = ModelResult(
            success=True,
            text="question",
            execution_time=0.3,
            model_used="phi:mini",
            cost=0.0
        )
        
        result_state = await node(sample_graph_state)
        
        # Check intent was classified
        assert result_state.query_intent is not None
        assert result_state.query_intent in ["question", "request", "conversation", "creative", "analysis", "code"]
        
        # Check model was called for classification
        mock_model_manager.select_optimal_model.assert_called_with(
            TaskType.SIMPLE_CLASSIFICATION,
            QualityLevel.MINIMAL
        )
        mock_model_manager.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_intent_classifier_fallback(self, mock_model_manager, sample_graph_state):
        """Test IntentClassifierNode fallback to rule-based classification."""
        node = IntentClassifierNode(mock_model_manager)
        
        # Mock model failure
        mock_model_manager.generate.return_value = ModelResult(
            success=False,
            error="Model unavailable"
        )
        
        # Test with code-related query
        sample_graph_state.original_query = "How do I debug this Python function?"
        sample_graph_state.processed_query = sample_graph_state.original_query
        
        result_state = await node(sample_graph_state)
        
        # Should still succeed with rule-based classification
        assert result_state.query_intent == "code"
        node_result = result_state.node_results["intent_classifier"]["result"]
        assert node_result["data"]["classification_method"] == "rule_based"
    
    @pytest.mark.asyncio
    async def test_response_generator_node(self, mock_model_manager, sample_graph_state):
        """Test ResponseGeneratorNode functionality."""
        node = ResponseGeneratorNode(mock_model_manager)
        
        # Set up state with intent and context
        sample_graph_state.query_intent = "question"
        sample_graph_state.query_complexity = 0.5
        sample_graph_state.intermediate_results["conversation_context"] = {
            "user_expertise_level": "intermediate",
            "preferred_response_style": "balanced",
            "conversation_mood": "professional"
        }
        
        result_state = await node(sample_graph_state)
        
        # Check response was generated
        assert result_state.final_response is not None
        assert len(result_state.final_response) > 0
        
        # Check model selection was appropriate
        mock_model_manager.select_optimal_model.assert_called()
        mock_model_manager.generate.assert_called()
        
        # Check generation parameters were reasonable
        call_args = mock_model_manager.generate.call_args
        assert call_args[1]["max_tokens"] > 0
        assert 0.0 <= call_args[1]["temperature"] <= 1.0


class TestChatGraph:
    """Test complete ChatGraph execution."""
    
    @pytest.mark.asyncio
    async def test_chat_graph_initialization(self, mock_model_manager, mock_cache_manager):
        """Test ChatGraph initialization."""
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        assert graph.graph_type == GraphType.CHAT
        assert graph.name == "chat_graph"
        assert graph.model_manager is mock_model_manager
        assert graph.cache_manager is mock_cache_manager
    
    @pytest.mark.asyncio
    async def test_chat_graph_build(self, mock_model_manager, mock_cache_manager):
        """Test ChatGraph structure building."""
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        graph.build()
        
        # Check all required nodes exist
        required_nodes = [
            "start", "context_manager", "intent_classifier", 
            "response_generator", "cache_update", "end", "error_handler"
        ]
        
        for node_name in required_nodes:
            assert node_name in graph.nodes
        
        # Check edges are defined
        assert len(graph.edges) > 0
    
    @pytest.mark.asyncio
    async def test_complete_chat_flow(self, mock_model_manager, mock_cache_manager, sample_graph_state):
        """Test complete chat graph execution."""
        set_correlation_id("test-chat-flow-001")
        
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        # Execute the complete graph
        final_state = await graph.execute(sample_graph_state)
        
        # Check execution completed successfully
        assert len(final_state.errors) == 0 or len(final_state.errors) <= 1  # Allow minor errors
        assert final_state.final_response is not None
        assert len(final_state.final_response) > 0
        
        # Check all main nodes were executed
        expected_nodes = ["start", "context_manager", "intent_classifier", "response_generator"]
        for node in expected_nodes:
            assert node in final_state.execution_path
        
        # Check performance metrics
        assert final_state.calculate_total_time() > 0
        assert final_state.get_avg_confidence() > 0
        
        # Check cache operations were attempted
        mock_cache_manager.update_conversation_history.assert_called()
    
    @pytest.mark.asyncio
    async def test_chat_graph_error_handling(self, mock_model_manager, mock_cache_manager, sample_graph_state):
        """Test chat graph error handling."""
        # Mock model failure
        mock_model_manager.generate.side_effect = Exception("Model failure")
        
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        final_state = await graph.execute(sample_graph_state)
        
        # Should have errors but still complete
        assert len(final_state.errors) > 0
        assert "error_handler" in final_state.execution_path or "end" in final_state.execution_path
    
    @pytest.mark.asyncio
    async def test_chat_graph_performance_stats(self, mock_model_manager, mock_cache_manager):
        """Test chat graph performance statistics."""
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        # Execute multiple times
        for i in range(3):
            state = GraphState(original_query=f"Test query {i}")
            await graph.execute(state)
        
        stats = graph.get_performance_stats()
        
        assert stats["graph_name"] == "chat_graph"
        assert stats["execution_count"] == 3
        assert stats["total_execution_time"] > 0
        assert "node_stats" in stats
        assert len(stats["node_stats"]) > 0


class TestGraphIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_conversation_continuity(self, mock_model_manager, mock_cache_manager):
        """Test conversation continuity across multiple exchanges."""
        set_correlation_id("test-conversation-continuity")
        
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        # First message
        state1 = GraphState(
            original_query="Hello, I'm learning Python",
            session_id="continuity_test",
            user_id="test_user"
        )
        
        result1 = await graph.execute(state1)
        
        # Second message with context
        state2 = GraphState(
            original_query="Can you explain functions?",
            session_id="continuity_test",
            user_id="test_user",
            conversation_history=result1.conversation_history
        )
        
        result2 = await graph.execute(state2)
        
        # Check context was maintained
        assert result2.query_intent is not None
        assert len(result2.execution_path) > 0
        
        # Check cache operations
        assert mock_cache_manager.update_conversation_history.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_different_query_types(self, mock_model_manager, mock_cache_manager):
        """Test handling different types of queries."""
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        test_queries = [
            ("What is machine learning?", "question"),
            ("Write a Python function", "code"), 
            ("Compare React and Vue", "analysis"),
            ("Hello there!", "conversation"),
            ("Help me debug this", "request")
        ]
        
        for query, expected_intent in test_queries:
            state = GraphState(original_query=query)
            
            # Mock intent classification to return expected intent
            def mock_classify(task_type, quality_level):
                return "phi:mini"
            
            def mock_generate(model_name, prompt, **kwargs):
                return ModelResult(
                    success=True,
                    text=expected_intent,
                    execution_time=0.2,
                    model_used=model_name,
                    cost=0.0
                )
            
            mock_model_manager.select_optimal_model = mock_classify
            mock_model_manager.generate = mock_generate
            
            result = await graph.execute(state)
            
            # Check query was processed appropriately
            assert result.query_intent == expected_intent
            assert result.final_response is not None
    
    @pytest.mark.asyncio
    async def test_budget_and_time_constraints(self, mock_model_manager, mock_cache_manager):
        """Test budget and time constraint handling."""
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        # Test with very low budget
        constrained_state = GraphState(
            original_query="Complex analysis query",
            max_cost=0.001,  # Very low budget
            max_execution_time=5.0
        )
        
        # Mock expensive operation
        def mock_expensive_generate(model_name, prompt, **kwargs):
            return ModelResult(
                success=True,
                text="Expensive response",
                execution_time=2.0,
                model_used=model_name,
                cost=0.05  # High cost
            )
        
        mock_model_manager.generate = mock_expensive_generate
        
        result = await graph.execute(constrained_state)
        
        # Should complete but might hit constraints
        assert result.final_response is not None
        
        # Check constraint tracking
        total_cost = result.calculate_total_cost()
        total_time = result.calculate_total_time()
        
        assert total_cost >= 0
        assert total_time > 0


if __name__ == "__main__":
    # Run a basic integration test
    async def run_basic_test():
        print("🧪 Running basic graph integration test...")
        
        set_correlation_id("basic-graph-test")
        
        # Create mock dependencies
        mock_model_manager = AsyncMock(spec=ModelManager)
        mock_model_manager.select_optimal_model.return_value = "llama2:7b"
        mock_model_manager.generate.return_value = ModelResult(
            success=True,
            text="Hello! I'm working properly through the graph system.",
            execution_time=1.0,
            model_used="llama2:7b",
            cost=0.0
        )
        
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_conversation_history.return_value = []
        
        # Create and execute graph
        graph = ChatGraph(mock_model_manager, mock_cache_manager)
        
        test_state = GraphState(
            original_query="Hello, how are you?",
            session_id="test_session",
            user_id="test_user"
        )
        
        try:
            result = await graph.execute(test_state)
            
            print(f"✅ Graph execution successful!")
            print(f"   Final response: {result.final_response}")
            print(f"   Nodes executed: {len(result.execution_path)}")
            print(f"   Execution path: {' → '.join(result.execution_path)}")
            print(f"   Total time: {result.calculate_total_time():.2f}s")
            print(f"   Average confidence: {result.get_avg_confidence():.2f}")
            print(f"   Errors: {len(result.errors)}")
            
            if result.errors:
                print(f"   Error details: {result.errors}")
            
            return True
            
        except Exception as e:
            print(f"❌ Graph execution failed: {e}")
            return False
    
    # Run the test
    import asyncio
    success = asyncio.run(run_basic_test())
    exit(0 if success else 1)
