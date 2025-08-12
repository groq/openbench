"""Test the registry module functionality."""

import pytest
from openbench.config import load_task, TASK_REGISTRY


def test_task_registry_contents():
    """Test that the task registry contains expected benchmarks."""
    assert "mmlu" in TASK_REGISTRY
    assert TASK_REGISTRY["mmlu"] == "openbench.evals.mmlu.mmlu"


def test_load_task_valid():
    """Test loading a valid task from the registry."""
    task = load_task("mmlu")
    assert callable(task)


def test_load_task_invalid():
    """Test loading an invalid task from the registry."""
    with pytest.raises(ValueError) as exc_info:
        load_task("nonexistent_benchmark")

    # Check that error message mentions available benchmarks
    assert "Unknown benchmark" in str(exc_info.value)
    assert "mmlu" in str(exc_info.value)


def test_load_task_caching():
    """Test that the load_task function uses caching."""
    # Call twice and verify it's the same object (due to lru_cache)
    task1 = load_task("mmlu")
    task2 = load_task("mmlu")
    assert task1 is task2  # Same object due to caching
