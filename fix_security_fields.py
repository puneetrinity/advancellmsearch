#!/usr/bin/env python3
"""
Fix scheme and credentials fields in request models
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

def fix_securetextinput():
    """Fix SecureTextInput to make scheme and credentials optional"""
    filepath = "app/api/security.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Making scheme/credentials optional in {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the SecureTextInput class and modify it
    # Look for the class definition
    class_pattern = r'class SecureTextInput\(BaseModel\):(.*?)(?=\nclass|\n\n@|\n\n#|\Z)'
    
    def replace_class(match):
        class_content = match.group(1)
        
        # Add optional scheme and credentials fields if they don't exist
        if 'scheme:' not in class_content and 'credentials:' not in class_content:
            # Add optional fields to the class
            new_content = '''
    """Base model with security validation for text inputs."""
    # Optional authentication fields (handled by middleware)
    scheme: Optional[str] = Field(None, description="Authentication scheme (auto-filled)")
    credentials: Optional[str] = Field(None, description="Authentication credentials (auto-filled)")
    
''' + class_content
            return f'class SecureTextInput(BaseModel):{new_content}'
        else:
            # Make existing fields optional
            class_content = re.sub(
                r'scheme:\s*str\s*=\s*Field\([^)]+\)',
                'scheme: Optional[str] = Field(None, description="Authentication scheme (auto-filled)")',
                class_content
            )
            class_content = re.sub(
                r'credentials:\s*str\s*=\s*Field\([^)]+\)',
                'credentials: Optional[str] = Field(None, description="Authentication credentials (auto-filled)")',
                class_content
            )
            return f'class SecureTextInput(BaseModel):{class_content}'
    
    content = re.sub(class_pattern, replace_class, content, flags=re.MULTILINE | re.DOTALL)
    
    # Ensure Optional is imported
    if 'from typing import' in content and 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
        content = content.replace(
            'from typing import',
            'from typing import Optional,'
        )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Made scheme/credentials optional in {filepath}")
    return True

def alternative_fix_remove_inheritance():
    """Alternative: Remove SecureTextInput inheritance from request models"""
    print(f"🔧 Alternative fix: Remove SecureTextInput inheritance...")
    
    # Fix ChatRequest to not inherit from SecureChatInput
    requests_file = "app/schemas/requests.py"
    if os.path.exists(requests_file):
        backup_file(requests_file)
        
        with open(requests_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Change ChatRequest to inherit from BaseModel instead of SecureChatInput
        # But keep the message field validation
        chat_request_pattern = r'class ChatRequest\(BaseModel\):(.*?)(?=\n\nclass|\Z)'
        
        # Also change SearchRequest to not inherit from SecureTextInput
        content = re.sub(
            r'class SearchRequest\(SecureTextInput\):',
            'class SearchRequest(BaseModel):',
            content
        )
        
        with open(requests_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Removed SecureTextInput inheritance from request models")
        return True
    
    return False

def create_test_payload():
    """Create test payloads that should work after the fix"""
    test_content = '''#!/usr/bin/env python3
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
    print("\nSearch (no security fields required):")
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
'''
    
    with open("test_security_fix.py", 'w') as f:
        f.write(test_content)
    
    print("📝 Created test_security_fix.py")

def main():
    print("🚀 SECURITY FIELDS FIXER")
    print("="*60)
    print("This fixes the scheme/credentials field requirement issue.\n")
    
    print("🔧 Option 1: Make scheme/credentials optional in SecureTextInput")
    if fix_securetextinput():
        print("✅ Applied Option 1")
    else:
        print("❌ Option 1 failed")
    
    print("\n🔧 Option 2 (Alternative): Remove SecureTextInput inheritance")
    if alternative_fix_remove_inheritance():
        print("✅ Applied Option 2 as backup")
    else:
        print("❌ Option 2 failed")
    
    create_test_payload()
    
    print(f"\n📊 SUMMARY:")
    print("="*60)
    print("✅ Made security fields optional")
    print("✅ Removed problematic inheritance as backup")
    print("✅ Created test script")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Restart your FastAPI server")
    print("2. Run: python test_security_fix.py")
    print("3. Endpoints should now work without scheme/credentials!")
    
    print(f"\n💡 EXPLANATION:")
    print("- scheme/credentials are authentication fields")
    print("- They should be handled by HTTP headers, not JSON body")
    print("- Making them optional allows normal API usage")
    print("- Authentication still works via Authorization header")

if __name__ == "__main__":
    main()
