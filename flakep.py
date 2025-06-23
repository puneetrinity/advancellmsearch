#!/usr/bin/env python3
"""
URGENT: Fix critical syntax errors + automated style cleanup
These syntax errors are preventing your code from running!
"""

import os
import re
from pathlib import Path


def fix_critical_syntax_errors():
    """Fix syntax errors that prevent code from running."""
    
    print("🚨 FIXING CRITICAL SYNTAX ERRORS...")
    
    syntax_fixes = [
        {
            'file': 'tests/integration/test_api_integration.py',
            'line': 155,
            'issue': 'unmatched )',
            'description': 'Remove extra closing parenthesis'
        },
        {
            'file': 'tests/integration/test_api_integration_FINAL.py', 
            'line': 157,
            'issue': 'unexpected indent',
            'description': 'Fix indentation'
        },
        {
            'file': 'tests/integration/test_complete_integration.py',
            'line': 51,
            'issue': 'unexpected indent', 
            'description': 'Fix indentation'
        }
    ]
    
    for fix in syntax_fixes:
        file_path = fix['file']
        if os.path.exists(file_path):
            print(f"🔧 Fixing syntax error in {file_path}:{fix['line']}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Quick fix for common issues
                if fix['issue'] == 'unmatched )':
                    # Find and remove extra closing parenthesis
                    target_line = fix['line'] - 1  # Convert to 0-based index
                    if target_line < len(lines):
                        line = lines[target_line]
                        # Remove one extra closing parenthesis
                        if line.count(')') > line.count('('):
                            lines[target_line] = line.replace(')', '', 1)
                            print(f"  ✅ Removed extra ')' from line {fix['line']}")
                
                elif fix['issue'] == 'unexpected indent':
                    # Fix indentation by removing extra spaces
                    target_line = fix['line'] - 1
                    if target_line < len(lines):
                        line = lines[target_line]
                        # Remove leading whitespace and re-indent properly
                        content = line.lstrip()
                        if content:
                            # Standard 4-space indent for function content
                            lines[target_line] = '    ' + content
                            print(f"  ✅ Fixed indentation on line {fix['line']}")
                
                # Write back the fixed file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                    
            except Exception as e:
                print(f"  ❌ Error fixing {file_path}: {e}")
        else:
            print(f"  ⚠️  File not found: {file_path}")


def clean_imports_and_variables(file_path):
    """Remove unused imports and variables."""
    print(f"🧹 Cleaning {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove specific unused imports
        unused_imports = [
            "from typing import AsyncGenerator",
            "from typing import List", 
            "from typing import Set",
            "from pydantic import BaseModel",
            "from pydantic import Field",
            "from pydantic import field_validator",
            "from app.core.async_utils import safe_execute",
            "from app.schemas.responses import StreamingChatResponse",
            "from app.schemas.responses import create_success_response",
            "from app.core.config import get_settings",
            "from fastapi import Query",
            "from app.core.async_utils import coroutine_safe",
            "from fastapi.staticfiles import StaticFiles",
            "from app.api.security import auth_stub",
            "from app.core.logging import setup_logging",
            "from datetime import timedelta",
            "from app.core.logging import get_correlation_id",
            "import logging",
            "import os",
            "import json",
            "import pytest",
        ]
        
        lines = content.split('\n')
        cleaned_lines = []
        removed_imports = []
        
        for line in lines:
            line_stripped = line.strip()
            should_remove = False
            
            for unused in unused_imports:
                if line_stripped == unused:
                    should_remove = True
                    removed_imports.append(unused.split()[-1])
                    break
            
            if not should_remove:
                cleaned_lines.append(line)
        
        # Fix unused variables by prefixing with underscore
        new_content = '\n'.join(cleaned_lines)
        
        # Fix F841 unused variables
        unused_var_fixes = [
            (r"(\s+)e = ", r"\1_e = "),  # except Exception as e: -> except Exception as _e:
            (r"(\s+)stats = ", r"\1_stats = "),
            (r"(\s+)session = ", r"\1_session = "),
            (r"(\s+)original_generate = ", r"\1_original_generate = "),
            (r"(\s+)severity_marker = ", r"\1_severity_marker = "),
        ]
        
        for pattern, replacement in unused_var_fixes:
            new_content = re.sub(pattern, replacement, new_content)
        
        # Remove F811 duplicate imports (keep only first occurrence)
        lines = new_content.split('\n')
        seen_imports = set()
        final_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')) and 'import' in line:
                import_signature = line.strip()
                if import_signature not in seen_imports:
                    seen_imports.add(import_signature)
                    final_lines.append(line)
                # Skip duplicate imports
            else:
                final_lines.append(line)
        
        new_content = '\n'.join(final_lines)
        
        # Write cleaned content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        if removed_imports:
            print(f"  ✅ Removed {len(removed_imports)} unused imports")
        
    except Exception as e:
        print(f"  ❌ Error cleaning {file_path}: {e}")


def fix_line_lengths(file_path):
    """Fix some of the easier long line issues."""
    print(f"📏 Fixing line lengths in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        fixes_made = 0
        
        for i, line in enumerate(lines):
            if len(line.rstrip()) > 79:
                # Try some simple fixes
                
                # Fix 1: Break long string literals
                if '("' in line and line.count('"') == 2:
                    # Simple string breaking
                    indent = len(line) - len(line.lstrip())
                    if 'logger.' in line or 'print(' in line:
                        # Break log messages and print statements
                        parts = line.split('("', 1)
                        if len(parts) == 2:
                            before = parts[0] + '('
                            after = '"' + parts[1]
                            if len(after) > 50:
                                # Split the string
                                string_content = after[1:-2]  # Remove quotes and )
                                if len(string_content) > 40:
                                    # Find a good break point
                                    break_point = string_content.find(' ', 30)
                                    if break_point > 0:
                                        part1 = string_content[:break_point]
                                        part2 = string_content[break_point:].strip()
                                        
                                        fixed_lines.append(before + '\n')
                                        fixed_lines.append(' ' * (indent + 4) + f'"{part1} "\n')
                                        fixed_lines.append(' ' * (indent + 4) + f'"{part2}")\n')
                                        fixes_made += 1
                                        continue
                
                # Fix 2: Break function calls with multiple parameters
                if '(' in line and ',' in line and ')' in line:
                    indent = len(line) - len(line.lstrip())
                    paren_start = line.find('(')
                    paren_end = line.rfind(')')
                    
                    if paren_start > 0 and paren_end > paren_start:
                        before_paren = line[:paren_start + 1]
                        params = line[paren_start + 1:paren_end]
                        after_paren = line[paren_end:]
                        
                        if ',' in params and len(params) > 30:
                            param_list = [p.strip() for p in params.split(',')]
                            if len(param_list) > 2:
                                fixed_lines.append(before_paren.rstrip() + '(\n')
                                for j, param in enumerate(param_list):
                                    if j == len(param_list) - 1:
                                        fixed_lines.append(' ' * (indent + 4) + param + '\n')
                                    else:
                                        fixed_lines.append(' ' * (indent + 4) + param + ',\n')
                                fixed_lines.append(' ' * indent + after_paren)
                                fixes_made += 1
                                continue
            
            # If no fix applied, keep original line
            fixed_lines.append(line)
        
        if fixes_made > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print(f"  ✅ Fixed {fixes_made} long lines")
        
    except Exception as e:
        print(f"  ❌ Error fixing line lengths in {file_path}: {e}")


def fix_test_decorators(file_path):
    """Fix E304 blank lines after function decorators in test files."""
    if 'test_' not in file_path:
        return
        
    print(f"🧪 Fixing test decorators in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a decorator
            if line.strip().startswith('@'):
                fixed_lines.append(line)
                
                # Look ahead to find the function definition
                j = i + 1
                while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('@')):
                    if lines[j].strip() != '':  # Keep other decorators
                        fixed_lines.append(lines[j])
                    j += 1
                
                # Now j should point to the function definition
                if j < len(lines) and (lines[j].strip().startswith('def ') or lines[j].strip().startswith('async def ')):
                    # Add the function definition without extra blank lines
                    fixed_lines.append(lines[j])
                    i = j
                else:
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        print(f"  ✅ Fixed decorator spacing")
        
    except Exception as e:
        print(f"  ❌ Error fixing decorators in {file_path}: {e}")


def main():
    """Run all fixes in priority order."""
    
    print("🚀 URGENT FLAKE8 FIXES - PRIORITY ORDER")
    print("="*50)
    
    # STEP 1: Fix critical syntax errors first
    fix_critical_syntax_errors()
    
    print("\n" + "="*50)
    print("🧹 CLEANING UP STYLE ISSUES")
    
    # STEP 2: Clean up the most problematic files
    priority_files = [
        'app/main.py',
        'app/api/chat.py', 
        'app/api/search.py',
        'app/api/security.py',
        'app/models/manager.py',
        'app/models/ollama_client.py',
    ]
    
    for file_path in priority_files:
        if os.path.exists(file_path):
            print(f"\n📁 Processing {file_path}")
            clean_imports_and_variables(file_path)
            fix_line_lengths(file_path)
    
    # STEP 3: Fix test files
    test_files = [
        'tests/integration/test_model_integration.py',
        'tests/integration/corrected_system_test.py',
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 Processing test file {file_path}")
            clean_imports_and_variables(file_path)
            fix_test_decorators(file_path)
    
    # STEP 4: Remove blank lines at end of files
    all_files = []
    for pattern in ['app/**/*.py', 'tests/**/*.py']:
        all_files.extend(Path('.').glob(pattern))
    
    for file_path in all_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove trailing blank lines and ensure single newline
                content = content.rstrip() + '\n'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except:
                pass
    
    print("\n" + "="*50)
    print("🎉 FIXES COMPLETED!")
    print("\nNext steps:")
    print("1. Run: flake8 app/ tests/ --max-line-length=88 --count")
    print("2. Check remaining issues")
    print("3. Test your code: python -c 'import app.main; print(\"✅ Import successful\")'")
    print("4. Run tests: python -m pytest tests/ -v")


if __name__ == "__main__":
    main()


# MANUAL FIXES STILL NEEDED:
"""
After running this script, you'll still need to manually fix:

1. Very long lines in app/main.py (lines 180, 285, etc.)
2. Function redefinition in app/api/security.py:755
3. Indentation issues in some test files
4. Any remaining syntax errors

Example manual fixes:

BEFORE (app/main.py:180):
some_very_long_line_that_exceeds_136_characters_and_needs_manual_breaking(param1, param2, param3)

AFTER:
some_very_long_line_that_exceeds_136_characters_and_needs_manual_breaking(
    param1, param2, param3
)

BEFORE (app/api/security.py - duplicate function):
def get_current_user():  # First definition
    ...

def get_current_user():  # Second definition - REMOVE THIS

AFTER:
def get_current_user():  # Keep only one
    ...
"""