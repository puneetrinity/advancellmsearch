import pytest
from fastapi.testclient import TestClient
from app.main import app

RESEARCH_ENDPOINT = "/api/v1/research/deep-dive"

client = TestClient(app)

def test_research_deep_dive_success():
    payload = {
        "research_question": "What are the latest advancements in quantum computing?",
        "max_results": 5,
        "quality": "high",
        "budget": 10.0
    }
    headers = {"Authorization": "Bearer test-user-key"}  # Replace with a valid key if needed
    resp = client.post(RESEARCH_ENDPOINT, json=payload, headers=headers)
    if resp.status_code != 200:
        print("\n--- Error Details ---\n", resp.text)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "data" in data
    assert "metadata" in data

def test_research_deep_dive_invalid_payload():
    payload = {"max_results": 1}
    headers = {"Authorization": "Bearer test-user-key"}
    resp = client.post(RESEARCH_ENDPOINT, json=payload, headers=headers)
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data

def test_research_deep_dive_unauthorized():
    payload = {
        "research_question": "test",
        "max_results": 1,
        "quality": "low",
        "budget": 1.0
    }
    resp = client.post(RESEARCH_ENDPOINT, json=payload)
    assert resp.status_code in (401, 403, 422)
    data = resp.json()
    assert "detail" in data or "error_details" in data

def test_research_deep_dive_budget_limit():
    payload = {
        "research_question": "Explain the theory of relativity.",
        "max_results": 10,
        "quality": "high",
        "budget": 0.01
    }
    headers = {"Authorization": "Bearer test-user-key"}
    resp = client.post(RESEARCH_ENDPOINT, json=payload, headers=headers)
    assert resp.status_code in (400, 402, 200, 422)
    data = resp.json()
    if resp.status_code == 200:
        assert data["metadata"]["cost"] <= 0.01
    else:
        assert "error" in str(data) or "budget" in str(data)
