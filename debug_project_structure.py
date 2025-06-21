#!/usr/bin/env python3
"""
Script to check for duplicate imports and path issues
"""
import os
import sys
import importlib.util
from pathlib import Path

def find_python_files():
    """Find all Python files in the project"""
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__', 'venv', 'env', 'node_modules'))]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def check_for_duplicate_files():
    """Check for duplicate Python files with same name"""
    print("🔍 CHECKING FOR DUPLICATE FILES...")
    print("="*60)
    
    file_map = {}
    python_files = find_python_files()
    
    for filepath in python_files:
        filename = os.path.basename(filepath)
        if filename not in file_map:
            file_map[filename] = []
        file_map[filename].append(filepath)
    
    duplicates_found = False
    for filename, paths in file_map.items():
        if len(paths) > 1:
            duplicates_found = True
            print(f"⚠️  DUPLICATE: {filename}")
            for path in paths:
                print(f"   {path}")
    
    if not duplicates_found:
        print("✅ No duplicate Python files found")
    
    return file_map

def check_main_py():
    """Check main.py for router registrations"""
    print(f"\n🔍 CHECKING MAIN.PY ROUTER REGISTRATIONS...")
    print("="*60)
    
    main_files = []
    for root, dirs, files in os.walk("."):
        if "main.py" in files:
            main_files.append(os.path.join(root, "main.py"))
    
    if not main_files:
        print("❌ No main.py found!")
        return
    
    for main_file in main_files:
        print(f"📄 {main_file}:")
        try:
            with open(main_file, 'r') as f:
                content = f.read()
                
            # Look for router includes
            lines = content.split('\n')
            router_lines = []
            for i, line in enumerate(lines, 1):
                if 'include_router' in line:
                    router_lines.append((i, line.strip()))
                elif 'from app.api' in line:
                    router_lines.append((i, line.strip()))
                elif '@app.post' in line or '@app.get' in line:
                    router_lines.append((i, line.strip()))
            
            if router_lines:
                print("   Router-related lines:")
                for line_num, line in router_lines:
                    print(f"     {line_num:3d}: {line}")
            else:
                print("   No router registrations found")
                
        except Exception as e:
            print(f"   ❌ Error reading {main_file}: {e}")

def check_api_files():
    """Check API files for endpoint definitions"""
    print(f"\n🔍 CHECKING API FILES FOR ENDPOINT DEFINITIONS...")
    print("="*60)
    
    api_files = []
    for root, dirs, files in os.walk("./app/api"):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                api_files.append(os.path.join(root, file))
    
    for api_file in api_files:
        print(f"📄 {api_file}:")
        try:
            with open(api_file, 'r') as f:
                content = f.read()
                
            lines = content.split('\n')
            endpoint_lines = []
            for i, line in enumerate(lines, 1):
                if '@router.post' in line or '@router.get' in line:
                    endpoint_lines.append((i, line.strip()))
                elif 'embed=False' in line or 'embed=True' in line:
                    endpoint_lines.append((i, line.strip()))
                elif 'Body(' in line:
                    endpoint_lines.append((i, line.strip()))
            
            if endpoint_lines:
                print("   Endpoint definitions:")
                for line_num, line in endpoint_lines:
                    print(f"     {line_num:3d}: {line}")
            else:
                print("   No endpoint definitions found")
                
        except Exception as e:
            print(f"   ❌ Error reading {api_file}: {e}")

def check_current_working_directory():
    """Check if we're running from the right directory"""
    print(f"\n🔍 CHECKING WORKING DIRECTORY...")
    print("="*60)
    
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in a project root (has app/ directory)
    if os.path.exists("app"):
        print("✅ Found app/ directory")
    else:
        print("⚠️  No app/ directory found - are you in the project root?")
    
    # Check sys.path
    print(f"\nPython sys.path (first 5 entries):")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")

def check_pycache():
    """Check for stale __pycache__ files"""
    print(f"\n🔍 CHECKING FOR STALE __pycache__ FILES...")
    print("="*60)
    
    pycache_dirs = []
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache_dirs.append(os.path.join(root, "__pycache__"))
    
    if pycache_dirs:
        print(f"Found {len(pycache_dirs)} __pycache__ directories:")
        for cache_dir in pycache_dirs[:10]:  # Show first 10
            print(f"  {cache_dir}")
        if len(pycache_dirs) > 10:
            print(f"  ... and {len(pycache_dirs) - 10} more")
        
        print(f"\n💡 RECOMMENDATION: Clear cache with:")
        print(f"   find . -name '__pycache__' -type d -exec rm -rf {{}} +")
        print(f"   # OR on Windows:")
        print(f"   for /d /r . %d in (__pycache__) do @if exist \"%d\" rd /s /q \"%d\"")
    else:
        print("✅ No __pycache__ directories found")

def main():
    print("🚀 PROJECT STRUCTURE DEBUGGER")
    print("="*60)
    print("This script checks for common issues that cause endpoint conflicts\n")
    
    # Check for duplicate files
    check_for_duplicate_files()
    
    # Check main.py
    check_main_py()
    
    # Check API files
    check_api_files()
    
    # Check working directory
    check_current_working_directory()
    
    # Check for stale cache
    check_pycache()
    
    print(f"\n📊 NEXT STEPS")
    print("="*60)
    print("1. Run debug_endpoints.py to see which schema is expected")
    print("2. Clear __pycache__ if found")
    print("3. Restart the server completely")
    print("4. Check that you're editing the right files")

if __name__ == "__main__":
    main()
