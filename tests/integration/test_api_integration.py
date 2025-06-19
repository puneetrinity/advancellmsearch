# tests/integration/test_api_integration.py
"""
Integration tests for the full API
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skip(reason="Requires running services for integration test")
def test_full_chat_flow():
    """Test complete chat flow (requires running services)"""
    pass


@pytest.mark.skip(reason="Requires running services for integration test")
def test_health_with_services():
    """Test health check with actual services"""
    pass

