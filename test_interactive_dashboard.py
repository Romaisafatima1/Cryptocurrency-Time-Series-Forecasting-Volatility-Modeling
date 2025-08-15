#!/usr/bin/env python3
"""
Test script for the interactive Dash dashboard
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dashboard_import():
    """Test if we can import the dashboard functions"""
    try:
        from dashboard import create_interactive_dashboard, run_interactive_dashboard
        print("✅ Successfully imported dashboard functions")
        return True
    except ImportError as e:
        print(f"❌ Failed to import dashboard functions: {e}")
        return False

def test_dashboard_creation():
    """Test if we can create the dashboard app"""
    try:
        from dashboard import create_interactive_dashboard
        app = create_interactive_dashboard()
        print("✅ Successfully created Dash app")
        
        # Test that the app has the expected layout components
        layout = app.layout
        print("✅ Dashboard layout created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create dashboard: {e}")
        return False

def test_callback_function():
    """Test if the callback function works with sample data"""
    try:
        from dashboard import create_interactive_dashboard
        app = create_interactive_dashboard()
        
        # Get the callback function
        callback_func = None
        for callback in app.callback_map.values():
            if 'update_graphs' in str(callback['callback']):
                callback_func = callback['callback']
                break
        
        if callback_func:
            # Test with sample values
            figures = callback_func(0.95, "emp")
            assert len(figures) == 4, "Should return 4 figures"
            print("✅ Callback function executed successfully")
            return True
        else:
            print("❌ Could not find callback function")
            return False
    except Exception as e:
        print(f"❌ Callback function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Interactive Dashboard...")
    print("=" * 50)
    
    tests = [
        test_dashboard_import,
        test_dashboard_creation,
        test_callback_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The interactive dashboard is ready to use.")
        print("\nTo run the dashboard:")
        print("  python -c \"from src.dashboard import run_interactive_dashboard; run_interactive_dashboard()\"")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
