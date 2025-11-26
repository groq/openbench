"""Test that registry loading doesn't produce import errors from any optional dependencies
that are incorrectly imported globally.
"""

import pytest


def test_imports_for_optional_dependencies():
    """Test that optional dependencies don't produce import errors from registry loading. 
    
    To fix this, wrap optional imports in try-except blocks instead of importing globally.
    """
    try:
        # Load all evaluations through the registry
        import openbench._registry
        
    except (ImportError, ModuleNotFoundError) as e:
        error_msg = (
            f"Found unhandled optional import error: {e}\n\n"
            "Fix: Wrap optional imports in try-except blocks\n"
            "Example:\n"
            "  try:\n"
            "      import optional_package  # type: ignore[import-untyped,import-not-found]\n"
            "  except ImportError:\n"
            "      optional_package = None"
        )
        pytest.fail(error_msg)