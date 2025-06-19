# tests/test_graphs.py
"""
Test LangGraph functionality
"""

import pytest
from app.graphs.base import GraphState, NodeResult


def test_graph_state_creation():
    """Test graph state initialization"""
    state = GraphState(
        original_query="Test query",
        user_id="test_user"
    )
    
    assert state.original_query == "Test query"
    assert state.user_id == "test_user"
    assert len(state.execution_path) == 0
    assert state.calculate_total_cost() == 0.0


def test_node_result_creation():
    """Test node result creation"""
    result = NodeResult(
        success=True,
        confidence=0.8,
        data={"test": "data"},
        cost=0.001
    )
    
    assert result.success is True
    assert result.confidence == 0.8
    assert result.data["test"] == "data"
    assert result.cost == 0.001


def test_graph_state_execution_tracking():
    """Test execution step tracking"""
    state = GraphState(original_query="Test")
    
    result = NodeResult(
        success=True,
        confidence=0.9,
        cost=0.002
    )
    
    state.add_execution_step("test_node", result)
    
    assert "test_node" in state.execution_path
    assert state.confidence_scores["test_node"] == 0.9
    assert state.costs_incurred["test_node"] == 0.002
    assert state.calculate_total_cost() == 0.002
