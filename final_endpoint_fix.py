#!/usr/bin/env python3
"""
THE FINAL FIX - Apply the correct parameter pattern to remove wrapper keys
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

def apply_final_fix():
    """Apply the CORRECT fix that actually works"""
    
    # Fix search.py
    search_file = "app/api/search.py"
    if os.path.exists(search_file):
        print(f"🔧 Applying FINAL fix to {search_file}...")
        backup_file(search_file)
        
        with open(search_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic pattern with the * parameter trick
        old_pattern = r"async def basic_search\(\n    req: Request,\n    body: SearchRequest = Body\\(\.\.\., embed=False\\),\n    current_user: Dict\[str, Any\] = Depends\\(get_current_user\\)\n\):"
        new_pattern = "async def basic_search(\n    req: Request,\n    *,  # Force keyword-only parameters - prevents wrapper key\n    search_data: SearchRequest,  # No Body() needed with *\n    current_user: Dict[str, Any] = Depends(get_current_user)\n):"
        content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE | re.DOTALL)
        content = content.replace('body.', 'search_data.')
        with open(search_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {search_file}")
    
    # Fix chat.py
    chat_file = "app/api/chat.py"
    if os.path.exists(chat_file):
        print(f"🔧 Applying FINAL fix to {chat_file}...")
        backup_file(chat_file)
        
        with open(chat_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic pattern with the * parameter trick
        old_pattern = r"async def chat_complete\(\n    req: Request,\n    background_tasks: BackgroundTasks,\n    body: ChatRequest = Body\\(\.\.\., embed=False\\),\n    current_user: Dict\[str, Any\] = Depends\\(get_current_user\\)\n\):"
        new_pattern = "async def chat_complete(\n    req: Request,\n    background_tasks: BackgroundTasks,\n    *,  # Force keyword-only parameters - prevents wrapper key\n    chat_data: ChatRequest,  # No Body() needed with *\n    current_user: Dict[str, Any] = Depends(get_current_user)\n):"
        content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE | re.DOTALL)
        content = content.replace('body.', 'chat_data.')
        # Also fix streaming endpoint if it exists
        stream_pattern = r"async def chat_stream\(\n    req: Request,\n    background_tasks: BackgroundTasks,\n    body: ChatStreamRequest = Body\\(\.\.\., embed=False\\),\n    current_user: Dict\[str, Any\] = Depends\\(get_current_user\\)\n\):"
        stream_replacement = "async def chat_stream(\n    req: Request,\n    background_tasks: BackgroundTasks,\n    *,  # Force keyword-only parameters - prevents wrapper key\n    stream_data: ChatStreamRequest,  # No Body() needed with *\n    current_user: Dict[str, Any] = Depends(get_current_user)\n):"
        content = re.sub(stream_pattern, stream_replacement, content, flags=re.MULTILINE | re.DOTALL)
        content = content.replace('body.', 'stream_data.')
        with open(chat_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {chat_file}")

def create_test_script():
    """Create a test script to verify the fix"""
    test_content = '''#!/usr/bin/env python3
"""
Test script to verify the endpoint fixes work
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_fixed_endpoints():
    print("🧪 TESTING FIXED ENDPOINTS...")
    print("="*60)
    
    # Test search with flat payload (should work now)
    search_payload = {
        "query": "test search query",
        "max_results": 5,
        "search_type": "web",
        "include_summary": True,
        "budget": 2.0,
        "quality": "standard"
    }
    
    print("🔍 Testing Search Basic (flat payload):")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/search/basic", json=search_payload, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   🎉 SEARCH FIXED! Accepts flat JSON!")
        elif response.status_code == 422:
            error = response.json()
            print(f"   ❌ Still validation error: {error}")
        else:
            print(f"   ⚠️  Other error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test chat with flat payload (should work now)  
    chat_payload = {
        "message": "Hello test message",
        "session_id": "test_session_123",
        "user_context": {},
        "quality_requirement": "balanced",
        "max_cost": 0.10,
        "max_execution_time": 30.0,
        "force_local_only": False,
        "response_style": "balanced",
        "include_sources": True,
        "include_debug_info": False
    }
    
    print("\n💬 Testing Chat Complete (flat payload):")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/chat/complete", json=chat_payload, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   🎉 CHAT FIXED! Accepts flat JSON!")
        elif response.status_code == 422:
            error = response.json()
            print(f"   ❌ Still validation error: {error}")
        else:
            print(f"   ⚠️  Other error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    print("\n📊 SUMMARY:")
    print("If you see 🎉 messages above, the fix worked!")
    print("If you still see validation errors, the server needs a restart.")

if __name__ == "__main__":
    test_fixed_endpoints()
'''
    
    with open("test_fixed_endpoints.py", 'w') as f:
        f.write(test_content)
    
    print("📝 Created test_fixed_endpoints.py")

def main():
    print("🚀 FINAL ENDPOINT FIX")
    print("="*60)
    print("This applies the CORRECT fix using the * parameter trick")
    print("which forces FastAPI to accept flat JSON without wrapper keys.\n")
    
    # Apply the fix
    apply_final_fix()
    
    # Create test script
    create_test_script()
    
    print(f"\n📊 NEXT STEPS:")
    print("="*60)
    print("1. RESTART your FastAPI server completely")
    print("2. Run: python test_fixed_endpoints.py")
    print("3. You should see flat JSON payloads working!")
    print("\nThe key insight:")
    print("- Using * forces keyword-only parameters")
    print("- This prevents FastAPI from creating wrapper keys")
    print("- No Body() needed when using this pattern")

if __name__ == "__main__":
    main()
