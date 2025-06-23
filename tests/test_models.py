# tests/test_models.py
"""
Test model management
"""

import pytest


@pytest.mark.asyncio
async def test_model_manager_initialization(mock_model_manager):
    """Test model manager initialization"""
    await mock_model_manager.initialize()
    assert mock_model_manager.is_healthy()


@pytest.mark.asyncio
async def test_model_generation(mock_model_manager):
    """Test model text generation"""
    result = await mock_model_manager.generate(
        model_name="phi:mini", prompt="Test prompt"
    )

    assert result.success is True
    assert result.text == "Mock response from model"
    assert result.model_used == "phi:mini"
    assert result.cost >= 0
