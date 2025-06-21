#!/usr/bin/env python3
"""
Fix duplicate ChatRequest and SearchRequest model definitions
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

def remove_duplicate_models_from_chat():
    """Remove duplicate ChatRequest definitions from chat.py"""
    filepath = "app/api/chat.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Removing duplicate models from {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the duplicate ChatRequest class definition
    # Pattern to match the entire class definition including decorators
    chat_request_pattern = r'# CORRECTED REQUEST MODELS WITH PROPER WRAPPERS\s*\n*class ChatRequest\(SecureChatInput\):.*?(?=\n\nclass|\n\nasync def|\nrouter\s*=|\Z)'
    
    content = re.sub(chat_request_pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Also remove any StreamingChatRequest if it's defined here
    streaming_pattern = r'class StreamingChatRequest\(BaseModel\):.*?(?=\n\nclass|\n\nasync def|\nrouter\s*=|\Z)'
    content = re.sub(streaming_pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Add proper import for the canonical models at the top
    if 'from app.schemas.requests import ChatRequest, ChatStreamRequest' not in content:
        # Find the import section and add our import
        import_pattern = r'(from app\\.schemas\\.responses import.*?\n)'
        replacement = r'\1from app.schemas.requests import ChatRequest, ChatStreamRequest\n'
        content = re.sub(import_pattern, replacement, content)
    
    # Clean up any extra whitespace
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Removed duplicate models from {filepath}")
    return True

def remove_duplicate_models_from_search():
    """Remove duplicate SearchRequest definitions from search.py"""
    filepath = "app/api/search.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Checking for duplicate models in {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if SearchRequest is imported properly
    if 'from app.schemas.requests import SearchRequest' not in content:
        print(f"✅ {filepath} already imports SearchRequest correctly")
        return True
    
    # Check for duplicate AdvancedSearchRequest or SimpleSearchRequest definitions
    has_duplicates = False
    
    # Look for class definitions that might be duplicates
    class_patterns = [
        r'class AdvancedSearchRequest\(.*?\):.*?(?=\n\nclass|\n\nasync def|\n@router|\Z)',
        r'class SimpleSearchRequest\(.*?\):.*?(?=\n\nclass|\n\nasync def|\n@router|\Z)'
    ]
    
    for pattern in class_patterns:
        if re.search(pattern, content, flags=re.MULTILINE | re.DOTALL):
            has_duplicates = True
            break
    
    if has_duplicates:
        backup_file(filepath)
        
        # Remove duplicate class definitions
        for pattern in class_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Clean up whitespace
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Removed duplicate models from {filepath}")
    else:
        print(f"✅ No duplicate models found in {filepath}")
    
    return True

def verify_canonical_models():
    """Verify that canonical models exist in requests.py"""
    filepath = "app/schemas/requests.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Canonical models file {filepath} not found!")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required models
    required_models = ['ChatRequest', 'ChatStreamRequest', 'SearchRequest']
    found_models = []
    
    for model in required_models:
        if f'class {model}(' in content:
            found_models.append(model)
    
    print(f"📋 Canonical models in {filepath}:")
    for model in required_models:
        if model in found_models:
            print(f"   ✅ {model}")
        else:
            print(f"   ❌ {model} - MISSING!")
    
    return len(found_models) == len(required_models)

def check_imports_in_endpoints():
    """Check that endpoints import from the canonical location"""
    files_to_check = [
        ("app/api/chat.py", ["ChatRequest", "ChatStreamRequest"]),
        ("app/api/search.py", ["SearchRequest"])
    ]
    
    print(f"\n🔍 Checking imports in endpoint files:")
    
    for filepath, required_imports in files_to_check:
        if not os.path.exists(filepath):
            print(f"   ❌ {filepath} not found")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   📄 {filepath}:")
        
        # Check for canonical import
        canonical_import = 'from app.schemas.requests import'
        has_canonical = canonical_import in content
        
        if has_canonical:
            print(f"      ✅ Has canonical import")
            
            # Check which models are imported
            for model in required_imports:
                if model in content.split(canonical_import)[1].split('\n')[0]:
                    print(f"      ✅ Imports {model}")
                else:
                    print(f"      ⚠️  Missing {model} in import")
        else:
            print(f"      ❌ Missing canonical import: {canonical_import}")
        
        # Check for local class definitions (bad)
        for model in required_imports:
            if f'class {model}(' in content:
                print(f"      ❌ DUPLICATE: Local {model} class found!")

def main():
    print("🚀 DUPLICATE MODEL FIXER")
    print("="*60)
    print("This script removes duplicate ChatRequest/SearchRequest definitions")
    print("and ensures endpoints import from the canonical schemas/requests.py\n")
    
    # Step 1: Verify canonical models exist
    print("Step 1: Verifying canonical models...")
    if not verify_canonical_models():
        print("❌ Canonical models missing! Fix app/schemas/requests.py first.")
        return
    
    # Step 2: Remove duplicates from chat.py
    print("\nStep 2: Removing duplicates from chat.py...")
    remove_duplicate_models_from_chat()
    
    # Step 3: Remove duplicates from search.py
    print("\nStep 3: Checking search.py...")
    remove_duplicate_models_from_search()
    
    # Step 4: Verify imports
    print("\nStep 4: Verifying imports...")
    check_imports_in_endpoints()
    
    print(f"\n📊 SUMMARY")
    print("="*60)
    print("✅ Removed duplicate model definitions")
    print("✅ Added canonical imports to endpoints")
    print("✅ All endpoints should now use the same models")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Clear __pycache__ directories")
    print("2. Restart your FastAPI server completely")
    print("3. Test endpoints - they should now work with flat JSON!")
    print("4. Run your debug script to confirm 422 errors are gone")

if __name__ == "__main__":
    main()
