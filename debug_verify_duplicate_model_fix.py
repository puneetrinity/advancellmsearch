#!/usr/bin/env python3
"""
Verify that the duplicate model fix worked
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_fixed_schemas():
    """Test that endpoints now accept the correct schema"""
    print("🧪 TESTING FIXED SCHEMAS...")
    print("="*60)
    
    # Test 1: Chat endpoint with correct flat payload
    print("💬 Testing Chat Complete with flat payload:")
    chat_payload = {
        "message": "Hello, this is a test message after fixing duplicates",
        "session_id": "test_session_fixed",
        "user_context": {},
        "quality_requirement": "balanced",
        "max_cost": 0.10,
        "max_execution_time": 30.0,
        "force_local_only": False,
        "response_style": "balanced",
        "include_sources": True,
        "include_debug_info": False
    }
    
    print(f"   Request: {json.dumps(chat_payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/chat/complete", json=chat_payload, timeout=15)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   🎉 SUCCESS! Chat accepts flat JSON!")
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
        elif response.status_code == 422:
            print("   ❌ Still validation error:")
            error_data = response.json()
            if "detail" in error_data:
                for error in error_data["detail"]:
                    loc = error.get("loc", [])
                    msg = error.get("msg", "")
                    print(f"      {'.'.join(str(x) for x in loc)}: {msg}")
        else:
            print(f"   ⚠️  Other error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 2: Search endpoint with correct flat payload
    print(f"\n🔍 Testing Search Basic with flat payload:")
    search_payload = {
        "query": "test search after fixing duplicates",
        "max_results": 5,
        "search_type": "web",
        "include_summary": True,
        "budget": 2.0,
        "quality": "standard"
    }
    
    print(f"   Request: {json.dumps(search_payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/search/basic", json=search_payload, timeout=15)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   🎉 SUCCESS! Search accepts flat JSON!")
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
        elif response.status_code == 422:
            print("   ❌ Still validation error:")
            error_data = response.json()
            if "detail" in error_data:
                for error in error_data["detail"]:
                    loc = error.get("loc", [])
                    msg = error.get("msg", "")
                    print(f"      {'.'.join(str(x) for x in loc)}: {msg}")
        else:
            print(f"   ⚠️  Other error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 3: Get OpenAPI schema to verify it changed
    print(f"\n📋 Checking OpenAPI schema:")
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        if response.status_code == 200:
            schema = response.json()
            
            # Check chat endpoint schema
            chat_path = schema.get("paths", {}).get("/api/v1/chat/complete", {})
            if chat_path:
                post_schema = chat_path.get("post", {})
                request_body = post_schema.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_ref = json_schema.get("schema", {})
                
                print(f"   Chat schema: {schema_ref}")
                
                # Check if it still expects wrapper
                if "Body_" in str(schema_ref):
                    print("   ⚠️  Still has Body_ wrapper schema")
                else:
                    print("   ✅ Schema looks correct (no Body_ wrapper)")
            
            # Check search endpoint schema
            search_path = schema.get("paths", {}).get("/api/v1/search/basic", {})
            if search_path:
                post_schema = search_path.get("post", {})
                request_body = post_schema.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_ref = json_schema.get("schema", {})
                
                print(f"   Search schema: {schema_ref}")
                
                if "Body_" in str(schema_ref):
                    print("   ⚠️  Still has Body_ wrapper schema")
                else:
                    print("   ✅ Schema looks correct (no Body_ wrapper)")
        else:
            print(f"   ❌ Could not get OpenAPI schema: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking schema: {e}")

def main():
    print("🚀 DUPLICATE MODEL FIX VERIFIER")
    print("="*60)
    print("This script verifies that duplicate models were removed")
    print("and endpoints now accept flat JSON payloads.\n")
    
    test_fixed_schemas()
    
    print(f"\n📊 EXPECTED RESULTS:")
    print("="*60)
    print("✅ Chat and Search should return 200 status")
    print("✅ No more 'Field: body.body -> Field required' errors")
    print("✅ OpenAPI schema should not have Body_ wrapper references")
    print("\nIf you still see 422 errors:")
    print("1. Make sure the server was restarted AFTER running fix_duplicate_models.py")
    print("2. Clear all __pycache__ directories")
    print("3. Check that imports in chat.py and search.py are correct")

if __name__ == "__main__":
    main()
