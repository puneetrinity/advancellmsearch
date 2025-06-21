#!/usr/bin/env python3
"""
Test payloads after fixing scheme/credentials issue
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_after_security_fix():
    """Test endpoints after fixing security field requirements"""
    print("TESTING AFTER SECURITY FIX...")
    print("="*60)
    # Test 1: Chat without scheme/credentials (should work now)
    print("Chat (no security fields required):")
    chat_payload = {
        "message": "Hello after security fix",
        "session_id": "test_security_fix",
        "user_context": {},
        "quality_requirement": "balanced",
        "max_cost": 0.10,
        "max_execution_time": 30.0,
        "force_local_only": False,
        "response_style": "balanced",
        "include_sources": True,
        "include_debug_info": False
    }
    try:
        response = requests.post(f"{BASE_URL}/api/v1/chat/complete", json=chat_payload, timeout=15)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   SUCCESS! Chat works without security fields!")
        elif response.status_code == 422:
            error = response.json()
            print(f"   Validation error: {error}")
            if "scheme" in str(error) or "credentials" in str(error):
                print("   Still requires scheme/credentials - try alternative fix")
            else:
                print("   Different validation error - check other required fields")
        else:
            print(f"   Other status: {response.text[:200]}")
    except Exception as e:
        print(f"   Request failed: {e}")
    # Test 2: Search without scheme/credentials (should work now)
    print("Search (no security fields required):")
    search_payload = {
        "query": "test search after security fix",
        "max_results": 5,
        "search_type": "web",
        "include_summary": True,
        "budget": 2.0,
        "quality": "standard"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/v1/search/basic", json=search_payload, timeout=15)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   SUCCESS! Search works without security fields!")
        elif response.status_code == 422:
            error = response.json()
            print(f"   Validation error: {error}")
            if "scheme" in str(error) or "credentials" in str(error):
                print("   Still requires scheme/credentials - try alternative fix")
            else:
                print("   Different validation error - check other required fields")
        else:
            print(f"   Other status: {response.text[:200]}")
    except Exception as e:
        print(f"   Request failed: {e}")

if __name__ == "__main__":
    test_after_security_fix()
