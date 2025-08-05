from functools import lru_cache
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Callable
from openbench.config import get_all_benchmarks, BenchmarkMetadata


# Dynamically generate registry from config
def _generate_task_registry():
    """Generate task registry from config."""
    registry = {}
    for name, metadata in get_all_benchmarks().items():
        registry[name] = f"{metadata.module_path}.{metadata.function_name}"
    return registry


TASK_REGISTRY = _generate_task_registry()


def _import_module_from_path(path: Path) -> ModuleType:
    """
    Import a .py file or package directory as an anonymous module.
    """
    file_path = path
    if path.is_dir():
        file_path = path / "__init__.py"
        if not file_path.exists():
            raise ValueError(f"{path} is a directory but has no __init__.py")

    mod_name = f"_openbench_dyn_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)

    # For packages, set up proper package structure for relative imports
    if path.is_dir():
        module.__package__ = mod_name
        sys.modules[mod_name] = module

        # Pre-load submodules to support relative imports
        for submodule_file in path.glob("*.py"):
            if submodule_file.name != "__init__.py":
                submodule_name = submodule_file.stem
                submodule_full_name = f"{mod_name}.{submodule_name}"
                submodule_spec = importlib.util.spec_from_file_location(
                    submodule_full_name, str(submodule_file)
                )
                if submodule_spec and submodule_spec.loader:
                    submodule = importlib.util.module_from_spec(submodule_spec)
                    submodule.__package__ = mod_name
                    sys.modules[submodule_full_name] = submodule
                    submodule_spec.loader.exec_module(submodule)
    else:
        sys.modules[mod_name] = module

    spec.loader.exec_module(module)
    return module


@lru_cache()
def load_task(benchmark_name: str) -> Callable:
    """
    Loads a task by benchmark name using the registry or from a local path.

    Args:
        benchmark_name (str): The name of the benchmark or path to a local eval.

    Returns:
        Callable: The imported function object.

    Raises:
        ValueError: If the benchmark is not in the registry and not a valid path.
        ImportError: If the module cannot be imported.
        AttributeError: If the function does not exist in the module.
    """
    # Try registry first (registry names take precedence)
    import_path = TASK_REGISTRY.get(benchmark_name)
    if import_path:
        module_path, func_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    # Fallback to path-based loading
    path = Path(benchmark_name).expanduser()
    if path.exists():
        return _load_task_from_local_path(path)

    # Neither registry nor valid path
    raise ValueError(
        f"Unknown benchmark: '{benchmark_name}'. "
        f"Available benchmarks: {', '.join(TASK_REGISTRY.keys())}"
    )


def _load_task_from_local_path(path: Path) -> Callable:
    """
    Load a task from a local path containing __metadata__.

    Args:
        path: Path to a directory or .py file containing an eval

    Returns:
        Callable: The imported function object

    Raises:
        ValueError: If no valid __metadata__ is found
        AttributeError: If the function does not exist in the module
        ImportError: If the module cannot be imported
    """
    root_module = _import_module_from_path(path)
    metadata = getattr(root_module, "__metadata__", None)

    if not isinstance(metadata, BenchmarkMetadata):
        raise ValueError(f"{path} has no valid __metadata__")

    # Resolve module path relative to root module
    # For local evals, module_path is typically relative like "simpleqa.simpleqa"
    # We need to extract just the last part and combine with the root module name
    if metadata.module_path.startswith(root_module.__name__):
        full_module_name = metadata.module_path
    else:
        # For paths like "simpleqa.simpleqa", we want the last component "simpleqa"
        module_components = metadata.module_path.split(".")
        module_name = module_components[-1]  # Take the last component
        full_module_name = f"{root_module.__name__}.{module_name}"

    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{full_module_name}': {e}")

    try:
        return getattr(module, metadata.function_name)
    except AttributeError:
        raise AttributeError(
            f"Function '{metadata.function_name}' not found in module '{full_module_name}'"
        )


def get_eval_metadata(path_like: str) -> BenchmarkMetadata | None:
    """
    Best-effort extraction of __metadata__ for path-based evals.
    Returns None for registry-based benchmarks or when no metadata is present.
    """
    p = Path(path_like).expanduser()
    if not p.exists():
        return None

    try:
        module = _import_module_from_path(p)
        meta = getattr(module, "__metadata__", None)
        return meta if isinstance(meta, BenchmarkMetadata) else None
    except Exception:
        return None
