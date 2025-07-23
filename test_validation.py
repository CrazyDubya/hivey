"""
Simple test script to validate core functionality and imports.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing imports...")

    try:
        import config  # noqa: F401

        print("✓ config module imported successfully")
    except Exception as e:
        print(f"✗ config import failed: {e}")
        return False

    try:
        import models  # noqa: F401

        print("✓ models module imported successfully")
    except Exception as e:
        print(f"✗ models import failed: {e}")
        return False

    try:
        import error_handling  # noqa: F401

        print("✓ error_handling module imported successfully")
    except Exception as e:
        print(f"✗ error_handling import failed: {e}")
        return False

    try:
        import async_llm_clients  # noqa: F401

        print("✓ async_llm_clients module imported successfully")
    except Exception as e:
        print(f"✗ async_llm_clients import failed: {e}")
        return False

    try:
        import utils  # noqa: F401

        print("✓ utils module imported successfully")
    except Exception as e:
        print(f"✗ utils import failed: {e}")
        return False

    try:
        import llm_clients  # noqa: F401

        print("✓ llm_clients module imported successfully")
    except Exception as e:
        print(f"✗ llm_clients import failed: {e}")
        return False

    try:
        # Test if swarms can be imported (may fail due to missing env vars)
        import swarms  # noqa: F401

        print("✓ swarms module imported successfully")
    except Exception as e:
        print(f"✗ swarms import failed (may be expected): {e}")
        # Not a failure for this basic test

    return True


def test_config():
    """Test configuration module functionality."""
    print("\nTesting configuration...")

    try:
        from config import config

        # Test that config object exists
        assert hasattr(config, "DB_NAME")
        assert hasattr(config, "DEFAULT_LLM_MODEL")
        assert hasattr(config, "get_api_key_header_name")

        # Test method call
        header_name = config.get_api_key_header_name()
        assert header_name == "X-API-Key"

        print("✓ Configuration module working correctly")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    print("\nTesting basic functionality...")

    try:
        # Test utils functions that don't require external connections
        from utils import configure_logging

        configure_logging()
        print("✓ Logging configuration successful")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running Hivey Code Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_config,
        test_basic_functionality,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
