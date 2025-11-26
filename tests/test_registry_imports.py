"""Test that registry loading doesn't produce import errors from any optional dependencies
that are incorrectly imported globally.
"""

import subprocess
import sys
import pytest


@pytest.fixture(scope="function")
def clean_environment():
    """Ensure only base dependencies are installed by running uv sync."""
    # Run uv sync to reset to base dependencies only
    result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to run 'uv sync': {result.stderr}")
    yield
    # Note: We don't restore optional deps after the test since other tests
    # should be able to run with base dependencies


def test_imports_for_optional_dependencies(clean_environment):
    """Test that optional dependencies don't produce import errors from registry loading. 
    
    This test runs in a clean environment with only base dependencies (uv sync)
    to ensure optional imports are properly wrapped in try-except blocks.
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