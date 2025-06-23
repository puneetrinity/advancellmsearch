import json

import requests

BASE_URL = "http://localhost:8000"

# 1. Test /api/v1/search/basic
search_payload = {
    "request": {"query": "test query", "max_results": 1, "include_summary": True}
}
search_resp = requests.post(f"{BASE_URL}/api/v1/search/basic", json=search_payload)
print("/api/v1/search/basic:", search_resp.status_code, search_resp.text)

# 2. Test /api/v1/chat/complete with extended schema
chat_payload = {
    "request": {
        "message": "Hello, this is a test",
        "session_id": "test_session",
        "quality_requirement": "balanced",
        "max_cost": 0.10,
        "max_execution_time": 30.0,
        "force_local_only": False,
        "response_style": "balanced",
        "include_sources": True,
        "include_debug_info": False,
        "user_context": {},
    }
}
chat_resp = requests.post(f"{BASE_URL}/api/v1/chat/complete", json=chat_payload)
print("/api/v1/chat/complete:", chat_resp.status_code, chat_resp.text)

# 3. Test /api/v1/chat/stream with minimal valid payload
stream_payload = {
    "request": {"messages": [{"role": "user", "content": "Stream this!"}]}
}
stream_resp = requests.post(f"{BASE_URL}/api/v1/chat/stream", json=stream_payload)
print("/api/v1/chat/stream:", stream_resp.status_code, stream_resp.text)
