#!/usr/bin/env python3
"""
Test script to verify endpoints accept flat JSON (no wrapper keys)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_flat_json():
    print("TESTING FLAT JSON ACCEPTANCE...")
    print("="*60)
    # Test 1: Chat with flat JSON
    print("Chat Complete (flat JSON):")
    chat_payload = {
        "message": "Hello, testing flat JSON after definitive fix",
        "session_id": "test_flat_json",
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
            print("   SUCCESS! Chat accepts flat JSON!")
            data = response.json()
            print(f"   Response type: {type(data)}")
            print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        elif response.status_code == 422:
            print("   Validation error:")
            try:
                error_data = response.json()
                print(f"      Error: {error_data}")
            except:
                print(f"      Raw error: {response.text}")
        else:
            print(f"   Unexpected status: {response.text[:200]}")
    except Exception as e:
        print(f"   Request failed: {e}")
    # Test 2: Search with flat JSON
    print("\nSearch Basic (flat JSON):")
    search_payload = {
        "query": "test search with flat JSON after fix",
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
            print("   SUCCESS! Search accepts flat JSON!")
            data = response.json()
            print(f"   Response type: {type(data)}")
            print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        elif response.status_code == 422:
            print("   Validation error:")
            try:
                error_data = response.json()
                print(f"      Error: {error_data}")
            except:
                print(f"      Raw error: {response.text}")
        else:
            print(f"   Unexpected status: {response.text[:200]}")
    except Exception as e:
        print(f"   Request failed: {e}")
    # Test 3: Verify OpenAPI schema changed
    print("\nChecking OpenAPI Schema:")
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        if response.status_code == 200:
            schema = response.json()
            paths = schema.get("paths", {})
            chat_schema = paths.get("/api/v1/chat/complete", {}).get("post", {}).get("requestBody", {})
            search_schema = paths.get("/api/v1/search/basic", {}).get("post", {}).get("requestBody", {})
            print(f"   Chat schema: {chat_schema}")
            print(f"   Search schema: {search_schema}")
            if "Body_" in str(chat_schema) or "Body_" in str(search_schema):
                print("   Still has Body_ wrapper references")
            else:
                print("   No Body_ wrapper references found")
        else:
            print(f"   Could not get schema: {response.status_code}")
    except Exception as e:
        print(f"   Schema check failed: {e}")

if __name__ == "__main__":
    test_flat_json()
