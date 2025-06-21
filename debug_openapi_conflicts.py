#!/usr/bin/env python3
"""
Debug script to identify duplicate/conflicting endpoint registrations
"""
import requests
import json
from pprint import pprint

BASE_URL = "http://localhost:8000"

def check_openapi_schema():
    """Check what the OpenAPI schema actually expects"""
    print("🔍 CHECKING OPENAPI SCHEMA...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code != 200:
            print(f"❌ Could not get OpenAPI schema: {response.status_code}")
            return
            
        schema = response.json()
        paths = schema.get("paths", {})
        
        print("📋 AVAILABLE ENDPOINTS:")
        for path in sorted(paths.keys()):
            methods = list(paths[path].keys())
            print(f"  {path} -> {methods}")
        
        print(f"\n🎯 SPECIFIC ENDPOINT ANALYSIS:")
        
        # Check chat endpoints
        chat_endpoints = [path for path in paths if "chat" in path]
        for endpoint in chat_endpoints:
            print(f"\n📝 {endpoint}:")
            endpoint_data = paths[endpoint]
            if "post" in endpoint_data:
                post_data = endpoint_data["post"]
                request_body = post_data.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_info = json_schema.get("schema", {})
                
                print(f"   Request Schema: {schema_info}")
                
                # Check if it references a component
                if "$ref" in schema_info:
                    ref_path = schema_info["$ref"]
                    ref_name = ref_path.split("/")[-1]
                    components = schema.get("components", {}).get("schemas", {})
                    if ref_name in components:
                        print(f"   Referenced Schema ({ref_name}):")
                        pprint(components[ref_name], width=80, depth=3)
        
        # Check search endpoints  
        search_endpoints = [path for path in paths if "search" in path]
        for endpoint in search_endpoints:
            print(f"\n🔍 {endpoint}:")
            endpoint_data = paths[endpoint]
            if "post" in endpoint_data:
                post_data = endpoint_data["post"]
                request_body = post_data.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_info = json_schema.get("schema", {})
                
                print(f"   Request Schema: {schema_info}")
                
                if "$ref" in schema_info:
                    ref_path = schema_info["$ref"]
                    ref_name = ref_path.split("/")[-1]
                    components = schema.get("components", {}).get("schemas", {})
                    if ref_name in components:
                        print(f"   Referenced Schema ({ref_name}):")
                        pprint(components[ref_name], width=80, depth=3)
                        
    except Exception as e:
        print(f"❌ Error checking schema: {e}")

def test_all_variants():
    """Test all possible endpoint/schema combinations"""
    print(f"\n🧪 TESTING ALL ENDPOINT VARIANTS...")
    print("="*60)
    
    # Define test payloads
    chat_flat = {
        "message": "test message",
        "session_id": "test_session"
    }
    
    chat_wrapped = {
        "request": {
            "message": "test message", 
            "session_id": "test_session"
        }
    }
    
    search_flat = {
        "query": "test query",
        "max_results": 5
    }
    
    search_wrapped = {
        "request": {
            "query": "test query",
            "max_results": 5
        }
    }
    
    # Test combinations
    test_cases = [
        # Chat endpoints
        ("/api/v1/chat/complete", chat_flat, "Chat Complete - Flat"),
        ("/api/v1/chat/complete", chat_wrapped, "Chat Complete - Wrapped"),
        ("/api/v1/chat/chat", chat_flat, "Chat Chat - Flat"),
        ("/api/v1/chat/chat", chat_wrapped, "Chat Chat - Wrapped"),
        
        # Search endpoints
        ("/api/v1/search/basic", search_flat, "Search Basic - Flat"),
        ("/api/v1/search/basic", search_wrapped, "Search Basic - Wrapped"),
        ("/api/v1/search/test", search_flat, "Search Test - Flat"),
        ("/api/v1/search/test", search_wrapped, "Search Test - Wrapped"),
    ]
    
    for endpoint, payload, description in test_cases:
        try:
            print(f"\n🔬 Testing: {description}")
            print(f"   Endpoint: {endpoint}")
            print(f"   Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=10)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 422:
                error_data = response.json()
                print(f"   Validation Error Details:")
                if "detail" in error_data:
                    for error in error_data["detail"]:
                        loc = error.get("loc", [])
                        msg = error.get("msg", "")
                        print(f"     Field: {'.'.join(str(x) for x in loc)} -> {msg}")
            elif response.status_code == 200:
                print("   ✅ SUCCESS!")
                resp_data = response.json()
                print(f"   Response keys: {list(resp_data.keys())}")
            elif response.status_code == 404:
                print("   ❌ ENDPOINT NOT FOUND")
            else:
                print(f"   ⚠️  Other Status: {response.text[:200]}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    print("🚀 ENDPOINT CONFLICT DEBUGGER")
    print("="*60)
    print("This script will help identify exactly what's causing the 422 errors")
    print("and which endpoint schema is actually expected.\n")
    
    # Step 1: Check OpenAPI schema
    check_openapi_schema()
    
    # Step 2: Test all variants
    test_all_variants()
    
    print(f"\n📊 SUMMARY")
    print("="*60)
    print("Look for:")
    print("1. Which endpoints return 200 (these work)")
    print("2. Which endpoints return 404 (these don't exist)")
    print("3. Validation error details for 422s")
    print("4. Any discrepancies between OpenAPI schema and actual behavior")

if __name__ == "__main__":
    main()
