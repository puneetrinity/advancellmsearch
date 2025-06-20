# Quick test to check if monitoring.py can be imported
# Run this from your project root to diagnose the issue

import sys
import traceback

print("Testing monitoring.py import...")

try:
    # Test basic Python syntax
    print("1. Testing file compilation...")
    import py_compile
    py_compile.compile('app/performance/monitoring.py', doraise=True)
    print("   ✅ File compiles successfully")
    
    # Test module import
    print("2. Testing module import...")
    import app.performance.monitoring as monitoring_module
    print("   ✅ Module imports successfully")
    
    # Test class availability
    print("3. Testing PerformanceTracker availability...")
    if hasattr(monitoring_module, 'PerformanceTracker'):
        print("   ✅ PerformanceTracker found in module")
        
        # Test class instantiation
        print("4. Testing PerformanceTracker instantiation...")
        tracker = monitoring_module.PerformanceTracker()
        print("   ✅ PerformanceTracker instantiated successfully")
        print(f"   📊 Class type: {type(tracker)}")
        
    else:
        print("   ❌ PerformanceTracker NOT found in module")
        print(f"   Available attributes: {dir(monitoring_module)}")
        
except SyntaxError as e:
    print(f"   ❌ Syntax error in monitoring.py:")
    print(f"   Line {e.lineno}: {e.text}")
    print(f"   Error: {e.msg}")
    
except Exception as e:
    print(f"   ❌ Import failed with error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nDone testing.")
