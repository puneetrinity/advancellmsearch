#!/usr/bin/env python3
"""
Automated fix script for flake8 issues in test files.
Run: python fix_flake8.py
"""

import os
import re
from pathlib import Path


def fix_flake8_issues(file_path):
    """Fix common flake8 issues automatically."""
    print(f"🔧 Fixing {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    # Track unused imports to remove
    unused_imports = {
        'from unittest.mock import patch',
        'from asgi_lifespan import LifespanManager',
        'import pytest',
        'from app.graphs.chat_graph import ChatGraph',
        'from app.graphs.base import BaseGraph',
        'from app.graphs.base import NodeType',
        'from fastapi.testclient import TestClient',
        'from app.core.logging import get_correlation_id'
    }
    
    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line
        
        # Fix 1: Remove trailing whitespace
        line = line.rstrip()
        
        # Fix 2: Remove unused imports
        if any(unused in line for unused in unused_imports):
            # Check if import is actually used in file
            import_name = None
            if 'import' in line:
                if 'from' in line and 'import' in line:
                    import_name = line.split('import')[-1].strip().split(',')[0].strip()
                elif line.strip().startswith('import '):
                    import_name = line.replace('import ', '').strip().split('.')[0]
                
                if import_name and import_name not in content:
                    print(f"  ❌ Removing unused import: {line.strip()}")
                    i += 1
                    continue
        
        # Fix 3: Fix f-strings without placeholders
        if line.strip().startswith('f"') and '{' not in line:
            line = line.replace('f"', '"')
        elif line.strip().startswith("f'") and '{' not in line:
            line = line.replace("f'", "'")
        
        # Fix 4: Break long lines (simple cases)
        if len(line) > 79:
            # Fix long function calls
            if '(' in line and ')' in line and '=' not in line[:line.find('(')]:
                indent = len(line) - len(line.lstrip())
                func_part = line[:line.find('(') + 1]
                params = line[line.find('(') + 1:line.rfind(')')].strip()
                end_part = line[line.rfind(')'):]
                
                if ',' in params and len(params) > 40:
                    # Split parameters
                    param_list = [p.strip() for p in params.split(',')]
                    fixed_lines.append(func_part)
                    for j, param in enumerate(param_list):
                        if j == len(param_list) - 1:
                            fixed_lines.append(' ' * (indent + 4) + param)
                        else:
                            fixed_lines.append(' ' * (indent + 4) + param + ',')
                    fixed_lines.append(' ' * indent + end_part)
                    i += 1
                    continue
            
            # Fix long string literals
            if '"' in line and len(line) > 79:
                # Simple string breaking
                indent = len(line) - len(line.lstrip())
                if line.count('"') == 2:  # Simple string
                    start_quote = line.find('"')
                    end_quote = line.rfind('"')
                    before = line[:start_quote + 1]
                    string_content = line[start_quote + 1:end_quote]
                    after = line[end_quote:]
                    
                    if len(string_content) > 50:
                        # Break string at word boundaries
                        words = string_content.split(' ')
                        current_line = before
                        
                        for word in words:
                            if len(current_line + word + ' ') > 79:
                                fixed_lines.append(current_line + '"')
                                current_line = ' ' * (indent + 4) + '"' + word + ' '
                            else:
                                current_line += word + ' '
                        
                        fixed_lines.append(current_line.rstrip() + after)
                        i += 1
                        continue
        
        # Fix 5: Add blank lines before class/function definitions
        if (line.strip().startswith('class ') or 
            line.strip().startswith('def ') or 
            line.strip().startswith('async def ')):
            
            # Check if we need blank lines before
            prev_line = lines[i-1].strip() if i > 0 else ""
            is_indented = line.startswith('    ')
            
            if prev_line and not is_indented:
                # Top-level class/function needs 2 blank lines
                if i > 1 and lines[i-2].strip():  # No blank lines before
                    fixed_lines.append('')
                    fixed_lines.append('')
            elif prev_line and is_indented:
                # Method needs 1 blank line
                if prev_line:  # Previous line is not blank
                    fixed_lines.append('')
        
        # Fix 6: Remove duplicate imports
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            if line in fixed_lines:
                print(f"  ❌ Removing duplicate import: {line.strip()}")
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Fix 7: Remove blank line at end of file and ensure single newline
    while fixed_lines and not fixed_lines[-1].strip():
        fixed_lines.pop()
    
    # Write fixed content
    new_content = '\n'.join(fixed_lines) + '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  ✅ Fixed {file_path}")


def main():
    """Fix all test files."""
    test_dirs = ['tests', 'tests/integration']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file_path in Path(test_dir).glob('*.py'):
                if file_path.name != '__init__.py':
                    fix_flake8_issues(file_path)
    
    print("\n🎉 Flake8 fixes completed!")
    print("\nNext steps:")
    print("1. Run: flake8 tests/ --max-line-length=88")
    print("2. Manually fix remaining issues if any")
    print("3. Run: python -m pytest tests/ -v")
    print("\nPython version recommendation:")
    print("- Use Python 3.10.x for best compatibility with Pydantic 1.x, langchain, and langgraph.")
    print("\nEnvironment setup options:")
    print("Option A: Use pyenv (Linux/Mac)")
    print("    pyenv install 3.10.12\n    pyenv local 3.10.12")
    print("Option B: Use conda (Windows/Mac/Linux)")
    print("    conda create -n llmsearch python=3.10.12\n    conda activate llmsearch")
    print("\nVerify fix:")
    print("    python --version  # Should show 3.10.x\n")


if __name__ == "__main__":
    main()
    