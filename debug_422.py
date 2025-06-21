#!/usr/bin/env python3
"""
Debug script to identify the exact cause of 422 validation errors.
This will tell us exactly what's wrong with the request validation.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint_with_debug(endpoint_path, payload, description):
    """Test an endpoint and show detailed error information"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint_path}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint_path}", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 422:
            print("❌ VALIDATION ERROR (422)")
            try:
                error_details = response.json()
                print("Error Details:")
                print(json.dumps(error_details, indent=2))
                
                # Extract specific validation errors
                if "detail" in error_details:
                    print("\nValidation Issues:")
                    for error in error_details["detail"]:
                        if isinstance(error, dict):
                            location = " -> ".join(str(loc) for loc in error.get("loc", []))
                            message = error.get("msg", "Unknown error")
                            error_type = error.get("type", "Unknown type")
                            print(f"  • {location}: {message} (type: {error_type})")
                        else:
                            print(f"  • {error}")
                            
            except Exception as e:
                print(f"Could not parse error response: {e}")
                print(f"Raw response: {response.text}")
                
        elif response.status_code == 200:
            print("✅ SUCCESS (200)")
            try:
                data = response.json()
                print(f"Response keys: {list(data.keys())}")
                if "status" in data:
                    print(f"Status: {data['status']}")
            except:
                print("Response is not JSON")
                
        else:
            print(f"⚠️ OTHER STATUS: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ REQUEST FAILED: {e}")

def main():
    """Run all tests to debug the 422 errors"""
    
    print("🔍 DEBUGGING 422 VALIDATION ERRORS")
    print("This script will test different request formats to identify the issue.")
    
    # Test 1: Search endpoint with flat format (now correct)
    test_endpoint_with_debug(
        "/api/v1/search/basic",
        {
            "query": "test search query",
            "max_results": 5,
            "include_summary": True
        },
        "Search with flat format (now correct)"
    )
    
    # Test 2: Chat endpoint with flat format (now correct)
    test_endpoint_with_debug(
        "/api/v1/chat/complete",
        {
            "message": "Hello test",
            "session_id": "test_session"
        },
        "Chat with flat format (now correct)"
    )
    
    # Test 3: Check if WrappedChatRequest is actually expected
    print(f"\n{'='*60}")
    print("CHECKING AVAILABLE ENDPOINTS")
    print(f"{'='*60}")
    
    try:
        # Get OpenAPI schema to see what the endpoints actually expect
        schema_response = requests.get(f"{BASE_URL}/openapi.json")
        if schema_response.status_code == 200:
            schema = schema_response.json()
            
            # Check chat endpoint schema
            chat_path = schema.get("paths", {}).get("/api/v1/chat/complete", {})
            if chat_path:
                post_schema = chat_path.get("post", {})
                request_body = post_schema.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_ref = json_schema.get("schema", {})
                
                print(f"Chat endpoint expects: {schema_ref}")
                
            # Check search endpoint schema  
            search_path = schema.get("paths", {}).get("/api/v1/search/basic", {})
            if search_path:
                post_schema = search_path.get("post", {})
                request_body = post_schema.get("requestBody", {})
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {})
                schema_ref = json_schema.get("schema", {})
                
                print(f"Search endpoint expects: {schema_ref}")
                
        else:
            print(f"Could not get OpenAPI schema: {schema_response.status_code}")
            
    except Exception as e:
        print(f"Error checking schemas: {e}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("This debug output shows exactly what validation errors are occurring.")
    print("Look for the specific 'loc' (location) and 'msg' (message) in the 422 errors.")
    print("This will tell us exactly what field is missing or incorrectly formatted.")

if __name__ == "__main__":
    main()
