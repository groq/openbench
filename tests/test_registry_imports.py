"""Test that registry loading doesn't produce import errors from any optional dependencies
that are incorrectly imported globally.
"""

import os
import subprocess
import tempfile
import pytest


def test_imports_for_optional_dependencies():
    """Test that optional dependencies don't produce import errors from registry loading.

    Creates an isolated temporary venv with only base dependencies to ensure optional
    imports are properly wrapped in try-except blocks. The temp venv is automatically
    deleted after the test.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "test_venv")
        python_exe = os.path.join(venv_path, "bin", "python")

        # Create temp venv and install package with only base dependencies
        steps = [
            (["uv", "venv", venv_path], "create venv"),
            (["uv", "pip", "install", "-e", ".", "--python", python_exe], "install openbench with base deps"),
        ]

        for cmd, desc in steps:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"Failed to {desc}: {result.stderr}")

        # Test the import in the venv
        result = subprocess.run(
            [python_exe, "-c", "import openbench._registry"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            pytest.fail(
                f"Found unhandled optional import error:\n{result.stderr}\n\n"
                "Fix: Wrap optional imports in try-except blocks\n"
                "Example:\n"
                "  try:\n"
                "      import optional_package  # type: ignore[import-untyped,import-not-found]\n"
                "  except ImportError:\n"
                "      optional_package = None"
            )