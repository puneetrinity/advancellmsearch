#!/usr/bin/env python3
"""
DEFINITIVE FIX: Replace endpoint signatures to eliminate Body wrapper requirement
This completely rewrites the problematic endpoint function signatures.
"""
import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modifying"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"📄 Backed up {filepath} to {backup_path}")
    return backup_path

def fix_chat_endpoint():
    """Completely rewrite chat endpoint to avoid Body wrapper"""
    filepath = "app/api/chat.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Completely rewriting chat endpoint in {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # NEW: Replace the entire chat_complete function signature
    old_signature = r'''@router\.post\("/complete", response_model=ChatResponse\)
@log_performance\("chat_complete"\)
@coroutine_safe\(timeout=60\.0\)
async def chat_complete\(
    req: Request,
    background_tasks: BackgroundTasks,
    request: ChatRequest = Body\(\.\.\., embed=False\),
    current_user: Dict\[str, Any\] = Depends\(get_current_user\)
\):'''

    new_signature = '''@router.post("/complete", response_model=ChatResponse)
@log_performance("chat_complete")
@coroutine_safe(timeout=60.0)
async def chat_complete(
    req: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Parse request body manually to avoid FastAPI wrapper
    try:
        body = await req.body()
        import json
        data = json.loads(body)
        chat_request = ChatRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {str(e)}")'''

    content = re.sub(old_signature, new_signature, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also fix streaming endpoint if it exists
    old_stream_signature = r'''@router\.post\("/stream".*?\)
[^)]*
async def chat_stream\(
    req: Request,
    background_tasks: BackgroundTasks,
    [^:]+: [^=]+ = Body\([^)]+\),
    current_user: Dict\[str, Any\] = Depends\(get_current_user\)
\):'''

    new_stream_signature = '''@router.post("/stream", response_class=StreamingResponse)
async def chat_stream(
    req: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Parse request body manually to avoid FastAPI wrapper
    try:
        body = await req.body()
        import json
        data = json.loads(body)
        stream_request = ChatStreamRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {str(e)}")'''

    content = re.sub(old_stream_signature, new_stream_signature, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add required imports if not present
    if 'import json' not in content:
        # Add after other imports
        import_pattern = r'(from app\.schemas\.responses import.*?\n)'
        replacement = r'\1import json\n'
        content = re.sub(import_pattern, replacement, content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed chat endpoint in {filepath}")
    return True

def fix_search_endpoint():
    """Completely rewrite search endpoint to avoid Body wrapper"""
    filepath = "app/api/search.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Completely rewriting search endpoint in {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the entire basic_search function signature
    old_signature = r'''@router\.post\("/basic", response_model=SearchResponse\)
@log_performance\("basic_search"\)
@coroutine_safe\(timeout=60\.0\)
async def basic_search\(
    req: Request,
    search_request: SearchRequest = Body\(\.\.\., embed=False\),
    current_user: Dict\[str, Any\] = Depends\(get_current_user\)
\):'''

    new_signature = '''@router.post("/basic", response_model=SearchResponse)
@log_performance("basic_search")
@coroutine_safe(timeout=60.0)
async def basic_search(
    req: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Parse request body manually to avoid FastAPI wrapper
    try:
        body = await req.body()
        import json
        data = json.loads(body)
        search_request = SearchRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {str(e)}")'''

    content = re.sub(old_signature, new_signature, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add required imports if not present
    if 'import json' not in content:
        # Add after other imports
        import_pattern = r'(from app\.schemas\.requests import SearchRequest\n)'
        replacement = r'\1import json\n'
        content = re.sub(import_pattern, replacement, content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed search endpoint in {filepath}")
    return True

def create_test_script():
    """Create a comprehensive test script"""
    test_content = '''#!/usr/bin/env python3
"""
Test script to verify endpoints accept flat JSON (no wrapper keys)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_flat_json():
    """Test that endpoints now accept flat JSON payloads"""
    print("🧪 TESTING FLAT JSON ACCEPTANCE...")
    print("="*60)
    
    # Test 1: Chat with flat JSON (should work now)
    print("💬 Testing Chat Complete (flat JSON):")
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
            print("   🎉 SUCCESS! Chat accepts flat JSON!")
            data = response.json()
            print(f"   Response type: {type(data)}")
            print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        elif response.status_code == 422:
            print("   ❌ Still validation error:")
            try:
                error_data = response.json()
                print(f"      Error: {error_data}")
            except:
                print(f"      Raw error: {response.text}")
        else:
            print(f"   ⚠️  Unexpected status: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 2: Search with flat JSON (should work now)
    print(f"\n🔍 Testing Search Basic (flat JSON):")
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
            print("   🎉 SUCCESS! Search accepts flat JSON!")
            data = response.json()
            print(f"   Response type: {type(data)}")
            print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        elif response.status_code == 422:
            print("   ❌ Still validation error:")
            try:
                error_data = response.json()
                print(f"      Error: {error_data}")
            except:
                print(f"      Raw error: {response.text}")
        else:
            print(f"   ⚠️  Unexpected status: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 3: Verify OpenAPI schema changed
    print(f"\n📋 Checking OpenAPI Schema:")
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
                print("   ⚠️  Still has Body_ wrapper references")
            else:
                print("   ✅ No Body_ wrapper references found")
        else:
            print(f"   ❌ Could not get schema: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Schema check failed: {e}")

if __name__ == "__main__":
    test_flat_json()
'''
    
    with open("test_definitive_fix.py", 'w') as f:
        f.write(test_content)
    
    print("📝 Created test_definitive_fix.py")

def main():
    print("🚀 DEFINITIVE ENDPOINT FIX")
    print("="*60)
    print("This completely rewrites endpoint signatures to parse JSON manually,")
    print("bypassing FastAPI's automatic Body parameter wrapping entirely.\n")
    
    success_count = 0
    
    # Fix chat endpoint
    if fix_chat_endpoint():
        success_count += 1
    
    # Fix search endpoint  
    if fix_search_endpoint():
        success_count += 1
    
    # Create test script
    create_test_script()
    
    print(f"\n📊 RESULTS:")
    print("="*60)
    print(f"✅ Fixed {success_count} endpoint files")
    print(f"✅ Created test script")
    
    print(f"\n🔧 CRITICAL NEXT STEPS:")
    print("="*60)
    print("1. RESTART your FastAPI server completely (kill process)")
    print("2. Run: python test_definitive_fix.py")
    print("3. Endpoints should now accept flat JSON without any wrapper!")
    print("\nThis fix bypasses FastAPI's Body() system entirely by:")
    print("- Removing all Body() parameters from function signatures")
    print("- Manually parsing request.body() as JSON")
    print("- Validating with Pydantic models after parsing")

if __name__ == "__main__":
    main()
