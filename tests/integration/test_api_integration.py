# tests/integration/test_api_integration.py
"""
Integration tests for the full API
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_search_basic():
    payload = {
        "query": "test query",
        "filters": {},
        "top_k": 3,
    }
    resp = client.post("/api/v1/search/basic", json=payload)
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_chat_complete():
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "graph_state": {},
        "quality": "default",
    }
    resp = client.post("/api/v1/chat/complete", json=payload)
    assert resp.status_code == 200
    assert "choices" in resp.json() or "result" in resp.json()


def test_search_test():
    payload = {"query": "test"}
    resp = client.post("/api/v1/search/test", json=payload)
    assert resp.status_code == 200
    assert "results" in resp.json() or resp.json().get("status") == "ok"

