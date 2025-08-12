"""
Evaluation configuration that combines static metadata with dynamic extraction.
Fast static data for list command, comprehensive dynamic data for describe.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

from openbench.config import get_benchmark_metadata, load_task


@dataclass
class EvalConfig:
    """Combined static metadata and dynamic config."""

    # Static metadata
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)

    # Dynamic data (populated on demand)
    epochs: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    sandbox: Optional[str] = None
    dataset_size: Optional[int] = None
    task_args: Optional[Dict[str, Any]] = None

    # Loading state
    _dynamic_loaded: bool = False


def _extract_task_config(task_name: str) -> Dict[str, Any]:
    """Extract configuration from actual task definition."""
    try:
        task_func = load_task(task_name)

        # Get task function signature to extract any arguments
        import inspect

        sig = inspect.signature(task_func)
        task_args = {}
        if sig.parameters:
            # Extract parameter names and defaults
            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    task_args[param_name] = param.default

        # Call task function to get the task object
        task = task_func()

        config = {
            "epochs": getattr(task, "epochs", None),
        }

        # Add task-specific arguments
        if task_args:
            config["task_args"] = task_args

        # Extract sandbox info
        sandbox = getattr(task, "sandbox", None)
        if sandbox:
            # Handle SandboxEnvironmentSpec or string
            if hasattr(sandbox, "type"):
                config["sandbox"] = sandbox.type
            else:
                config["sandbox"] = str(sandbox)
        else:
            config["sandbox"] = None

        # Extract all GenerateConfig values dynamically
        if hasattr(task, "config") and task.config:
            # Get all fields from GenerateConfig that have values
            for field_name in task.config.model_fields:
                value = getattr(task.config, field_name, None)
                if value is not None:
                    config[field_name] = value

        # Try to get dataset size without loading full dataset
        if hasattr(task, "dataset"):
            try:
                # Some datasets have length without loading
                config["dataset_size"] = len(task.dataset)
            except Exception:
                config["dataset_size"] = None

        return config
    except Exception:
        return {}


@lru_cache(maxsize=None)
def get_eval_config(name: str, load_dynamic: bool = False) -> Optional[EvalConfig]:
    """Get evaluation configuration by name.

    Args:
        name: Benchmark name
        load_dynamic: Whether to load dynamic data (slow but comprehensive)
    """
    metadata = get_benchmark_metadata(name)
    if not metadata:
        return None

    config = EvalConfig(
        name=metadata.name,
        description=metadata.description,
        category=metadata.category,
        tags=metadata.tags,
    )

    if load_dynamic:
        dynamic_data = _extract_task_config(name)
        config.epochs = dynamic_data.get("epochs")
        config.temperature = dynamic_data.get("temperature")
        config.max_tokens = dynamic_data.get("max_tokens")
        config.sandbox = dynamic_data.get("sandbox")
        config.dataset_size = dynamic_data.get("dataset_size")
        config.task_args = dynamic_data.get("task_args")
        config._dynamic_loaded = True

    return config
