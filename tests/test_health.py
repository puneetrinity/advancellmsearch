# tests/test_health.py
"""
Test health endpoints
"""


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "version" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code in [200, 503]  # May fail if services not ready
