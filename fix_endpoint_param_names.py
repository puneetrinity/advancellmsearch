#!/usr/bin/env python3
"""
Automatically fix endpoint parameter names to resolve FastAPI wrapper key issue
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

def fix_search_py():
    """Fix search.py endpoint parameter names"""
    filepath = "app/api/search.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Fixing {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the main search endpoint
    # Pattern: search_request: SearchRequest = Body(..., embed=False)
    # Replace with: body: SearchRequest = Body(..., embed=False)
    content = re.sub(
        r'(\s+)search_request:\s*SearchRequest\s*=\s*Body\([^)]+\)',
        r'\1body: SearchRequest = Body(..., embed=False)',
        content
    )
    
    # Fix all references to search_request in the function bodies
    function_patterns = [
        (r'\bsearch_request\.query\b', 'body.query'),
        (r'\bsearch_request\.budget\b', 'body.budget'),
        (r'\bsearch_request\.quality\b', 'body.quality'),
        (r'\bsearch_request\.max_results\b', 'body.max_results'),
        (r'\bsearch_request\.search_type\b', 'body.search_type'),
        (r'\bsearch_request\.include_summary\b', 'body.include_summary'),
        (r'\bsearch_request\.filters\b', 'body.filters'),
        (r'\bsearch_request\.date_range\b', 'body.date_range'),
        (r'\bsearch_request\.domains\b', 'body.domains'),
        (r'\bsearch_request\.language\b', 'body.language'),
        (r'\bsearch_request\.safe_search\b', 'body.safe_search'),
    ]
    
    for pattern, replacement in function_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Also fix any other search request parameters if they exist
    content = re.sub(
        r'(\s+)([a-zA-Z_]+_request):\s*([A-Za-z]+Request)\s*=\s*Body\([^)]+\)',
        r'\1body: \3 = Body(..., embed=False)',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed {filepath}")
    return True

def fix_chat_py():
    """Fix chat.py endpoint parameter names"""
    filepath = "app/api/chat.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    print(f"🔧 Fixing {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the main chat endpoint
    # Pattern: chat_request: ChatRequest = Body(..., embed=False)
    # Replace with: body: ChatRequest = Body(..., embed=False)
    content = re.sub(
        r'(\s+)chat_request:\s*ChatRequest\s*=\s*Body\([^)]+\)',
        r'\1body: ChatRequest = Body(..., embed=False)',
        content
    )
    
    # Fix streaming endpoint if it exists
    content = re.sub(
        r'(\s+)stream_request:\s*ChatStreamRequest\s*=\s*Body\([^)]+\)',
        r'\1body: ChatStreamRequest = Body(..., embed=False)',
        content
    )
    
    # Fix all references to chat_request in the function bodies
    function_patterns = [
        (r'\bchat_request\.message\b', 'body.message'),
        (r'\bchat_request\.session_id\b', 'body.session_id'),
        (r'\bchat_request\.user_context\b', 'body.user_context'),
        (r'\bchat_request\.quality_requirement\b', 'body.quality_requirement'),
        (r'\bchat_request\.max_cost\b', 'body.max_cost'),
        (r'\bchat_request\.max_execution_time\b', 'body.max_execution_time'),
        (r'\bchat_request\.force_local_only\b', 'body.force_local_only'),
        (r'\bchat_request\.response_style\b', 'body.response_style'),
        (r'\bchat_request\.include_sources\b', 'body.include_sources'),
        (r'\bchat_request\.include_debug_info\b', 'body.include_debug_info'),
        (r'\bchat_request\.context\b', 'body.context'),
        (r'\bchat_request\.constraints\b', 'body.constraints'),
        # For streaming requests
        (r'\bstream_request\.messages\b', 'body.messages'),
        (r'\bstream_request\.session_id\b', 'body.session_id'),
        (r'\bstream_request\.model\b', 'body.model'),
        (r'\bstream_request\.max_tokens\b', 'body.max_tokens'),
        (r'\bstream_request\.temperature\b', 'body.temperature'),
        (r'\bstream_request\.stream\b', 'body.stream'),
        (r'\bstream_request\.user_preferences\b', 'body.user_preferences'),
        (r'\bstream_request\.quality_requirement\b', 'body.quality_requirement'),
        (r'\bstream_request\.max_completion_time\b', 'body.max_completion_time'),
    ]
    
    for pattern, replacement in function_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Also fix any other chat request parameters if they exist
    content = re.sub(
        r'(\s+)([a-zA-Z_]+_request):\s*([A-Za-z]+Request)\s*=\s*Body\([^)]+\)',
        r'\1body: \3 = Body(..., embed=False)',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed {filepath}")
    return True

def verify_fixes():
    """Verify that fixes were applied correctly"""
    print(f"\n🔍 VERIFYING FIXES...")
    print("="*60)
    
    files_to_check = ["app/api/search.py", "app/api/chat.py"]
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"⚠️  {filepath} not found")
            continue
            
        print(f"📄 Checking {filepath}:")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for remaining problematic patterns
        problematic_patterns = [
            r'search_request:\s*SearchRequest\s*=\s*Body',
            r'chat_request:\s*ChatRequest\s*=\s*Body',
            r'stream_request:\s*ChatStreamRequest\s*=\s*Body',
        ]
        
        issues_found = []
        for pattern in problematic_patterns:
            matches = re.findall(pattern, content)
            if matches:
                issues_found.extend(matches)
        
        if issues_found:
            print(f"   ⚠️  Still has problematic patterns: {issues_found}")
        else:
            print(f"   ✅ No problematic patterns found")
        
        # Check for correct patterns
        good_patterns = [
            r'body:\s*SearchRequest\s*=\s*Body\([^)]*embed=False',
            r'body:\s*ChatRequest\s*=\s*Body\([^)]*embed=False',
            r'body:\s*ChatStreamRequest\s*=\s*Body\([^)]*embed=False',
        ]
        
        good_found = []
        for pattern in good_patterns:
            matches = re.findall(pattern, content)
            good_found.extend(matches)
        
        if good_found:
            print(f"   ✅ Found correct patterns: {len(good_found)}")
        else:
            print(f"   ⚠️  No correct patterns found - check if file has endpoints")

def main():
    print("🚀 FASTAPI ENDPOINT PARAMETER FIXER")
    print("="*60)
    print("This script fixes the parameter naming issue that causes")
    print("FastAPI to expect wrapper keys in request bodies.\n")
    
    # Check current directory
    if not os.path.exists("app/api"):
        print("❌ app/api directory not found. Run this from project root.")
        return
    
    success_count = 0
    
    # Fix search.py
    if fix_search_py():
        success_count += 1
    
    # Fix chat.py
    if fix_chat_py():
        success_count += 1
    
    # Verify fixes
    verify_fixes()
    
    print(f"\n📊 SUMMARY")
    print("="*60)
    print(f"✅ Fixed {success_count} files")
    print(f"\n🔧 NEXT STEPS:")
    print("1. Clear __pycache__ directories:")
    print("   for /d /r . %d in (__pycache__) do @if exist \"%d\" rd /s /q \"%d\"")
    print("2. Restart your FastAPI server completely")
    print("3. Test with flat JSON payloads (no wrapper keys)")
    print("4. Run your debug script to confirm 422 errors are gone")
    
    if success_count > 0:
        print(f"\n🎉 All endpoint parameter names have been fixed!")
        print(f"Your endpoints should now accept flat JSON payloads.")
    else:
        print(f"\n⚠️  No files were modified. Check file paths and try again.")

if __name__ == "__main__":
    main()
