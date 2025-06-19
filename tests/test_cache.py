# tests/test_cache.py
"""
Test cache functionality
"""

import pytest


@pytest.mark.asyncio
async def test_cache_basic_operations(mock_cache_manager):
    """Test basic cache operations"""
    await mock_cache_manager.initialize()
    
    # Test set and get
    await mock_cache_manager.set("test_key", {"data": "test_value"})
    result = await mock_cache_manager.get("test_key")
    
    assert result == {"data": "test_value"}
    
    # Test default value
    result = await mock_cache_manager.get("nonexistent_key", "default")
    assert result == "default"


@pytest.mark.asyncio
async def test_budget_tracking(mock_cache_manager):
    """Test budget tracking"""
    budget = await mock_cache_manager.get_remaining_budget("test_user")
    assert budget == 100.0


@pytest.mark.asyncio
async def test_rate_limiting(mock_cache_manager):
    """Test rate limiting"""
    allowed, count = await mock_cache_manager.check_rate_limit("test_user", 10)
    assert allowed is True
    assert count == 1


