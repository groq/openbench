"""Utilities for dynamic module imports."""

import os
import importlib.util


def import_module_from_same_dir(caller_file: str, module_name: str):
    """Import a module from the same directory as the caller file.

    Args:
        caller_file: The __file__ attribute of the calling module
        module_name: Name of the module to import (without .py extension)

    Returns:
        The imported module
    """
    current_dir = os.path.dirname(caller_file)
    module_path = os.path.join(current_dir, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
