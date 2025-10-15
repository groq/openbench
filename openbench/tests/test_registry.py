"""Test the registry module functionality."""

import pytest
from unittest.mock import Mock, patch
from openbench.config import (
    load_task,
    TASK_REGISTRY,
    _load_entry_point_benchmarks,
)
from openbench.utils import BenchmarkMetadata


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


def test_load_task_dash_underscore_equivalence():
    """Dash/underscore variants should load the same benchmark."""

    dash_task = load_task("mmlu-pro")
    underscore_task = load_task("mmlu_pro")
    assert dash_task is underscore_task


def test_load_task_dash_alias_for_underscore_name():
    """Benchmarks registered with underscores accept dash aliases."""

    task = load_task("gpqa-diamond")
    assert callable(task)


def test_load_task_alpha_requires_flag_with_dash_variant():
    """Alpha benchmarks still require --alpha even with dash aliases."""

    with pytest.raises(ValueError):
        load_task("graphwalks-parents")

    task = load_task("graphwalks-parents", allow_alpha=True)
    assert callable(task)


# Entry Point Tests


@patch("openbench.config.entry_points")
def test_load_entry_point_single_benchmark(mock_entry_points):
    """Test loading a single benchmark from an entry point."""
    # Create metadata that will be returned
    metadata = BenchmarkMetadata(
        name="Custom Benchmark",
        description="A custom benchmark from an external package",
        category="community",
        tags=["custom"],
        module_path="custom_pkg.benchmark",
        function_name="custom_benchmark",
    )

    # Create a mock function that returns the metadata
    mock_func = Mock(return_value=metadata)

    # Create a mock entry point
    mock_ep = Mock()
    mock_ep.name = "custom_benchmark"
    mock_ep.load.return_value = mock_func

    # Mock entry_points().select() to return our mock
    mock_select = Mock(return_value=[mock_ep])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify the benchmark was loaded
    assert "custom_benchmark" in result
    assert result["custom_benchmark"].name == "Custom Benchmark"
    assert result["custom_benchmark"].category == "community"
    # Verify the function was called
    mock_func.assert_called_once()


@patch("openbench.config.entry_points")
def test_load_entry_point_multiple_benchmarks(mock_entry_points):
    """Test loading multiple benchmarks from a single entry point."""
    # Create the dict that will be returned
    benchmarks_dict = {
        "benchmark_a": BenchmarkMetadata(
            name="Benchmark A",
            description="First benchmark",
            category="community",
            tags=["custom"],
            module_path="custom_pkg.benchmark_a",
            function_name="benchmark_a",
        ),
        "benchmark_b": BenchmarkMetadata(
            name="Benchmark B",
            description="Second benchmark",
            category="community",
            tags=["custom"],
            module_path="custom_pkg.benchmark_b",
            function_name="benchmark_b",
        ),
    }

    # Create a mock function that returns the dict
    mock_func = Mock(return_value=benchmarks_dict)

    # Create a mock entry point
    mock_ep = Mock()
    mock_ep.name = "custom_pkg_benchmarks"
    mock_ep.load.return_value = mock_func

    # Mock entry_points to return our mock
    mock_select = Mock(return_value=[mock_ep])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify both benchmarks were loaded
    assert "benchmark_a" in result
    assert "benchmark_b" in result
    assert result["benchmark_a"].name == "Benchmark A"
    assert result["benchmark_b"].name == "Benchmark B"
    # Verify the function was called
    mock_func.assert_called_once()


@patch("openbench.config.entry_points")
@patch("openbench.config.logger")
def test_load_entry_point_error_handling(mock_logger, mock_entry_points):
    """Test that entry point loading errors are handled gracefully."""
    # Create a mock entry point that raises an error
    mock_ep_error = Mock()
    mock_ep_error.name = "broken_benchmark"
    mock_ep_error.load.side_effect = ImportError("Module not found")

    # Create a valid entry point
    mock_ep_valid = Mock()
    mock_ep_valid.name = "valid_benchmark"
    mock_ep_valid.load.return_value = BenchmarkMetadata(
        name="Valid Benchmark",
        description="A valid benchmark",
        category="community",
        tags=["custom"],
        module_path="valid_pkg.benchmark",
        function_name="valid_benchmark",
    )

    # Mock entry_points().select() to return both
    mock_select = Mock(return_value=[mock_ep_error, mock_ep_valid])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify valid benchmark was loaded, broken one was skipped
    assert "valid_benchmark" in result
    assert "broken_benchmark" not in result

    # Check warning was logged
    mock_logger.warning.assert_called()
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any(
        "Failed to load benchmark from entry point 'broken_benchmark'" in call
        for call in warning_calls
    )


@patch("openbench.config.entry_points")
@patch("openbench.config.logger")
def test_load_entry_point_invalid_return_type(mock_logger, mock_entry_points):
    """Test handling of entry points that return invalid types."""
    # Create a mock entry point that returns invalid type
    mock_ep = Mock()
    mock_ep.name = "invalid_benchmark"
    mock_ep.load.return_value = "not a BenchmarkMetadata"

    # Mock entry_points().select() to return our mock
    mock_select = Mock(return_value=[mock_ep])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify nothing was loaded
    assert len(result) == 0

    # Check warning was logged
    mock_logger.warning.assert_called()
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("returned unexpected type" in call for call in warning_calls)


@patch("openbench.config.entry_points")
def test_load_entry_point_can_override_builtin(mock_entry_points):
    """Test that entry points can override built-in benchmarks."""
    # Create metadata for override
    metadata = BenchmarkMetadata(
        name="Custom MMLU",
        description="A custom version of MMLU",
        category="custom",
        tags=["custom"],
        module_path="custom_pkg.mmlu",
        function_name="custom_mmlu",
    )

    # Create a mock function that returns the metadata
    mock_func = Mock(return_value=metadata)

    # Create a mock entry point that overrides a built-in
    mock_ep = Mock()
    mock_ep.name = "mmlu"  # This is a built-in benchmark
    mock_ep.load.return_value = mock_func

    # Mock entry_points().select() to return our mock
    mock_select = Mock(return_value=[mock_ep])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify the entry point was loaded (can override)
    assert "mmlu" in result
    assert result["mmlu"].name == "Custom MMLU"
    assert result["mmlu"].category == "custom"
    mock_func.assert_called_once()


@patch("openbench.config.entry_points")
def test_load_entry_point_override_in_dict(mock_entry_points):
    """Test that dict entries can override built-in benchmarks."""
    # Create the dict with both custom and override
    benchmarks_dict = {
        "custom_bench": BenchmarkMetadata(
            name="Custom Benchmark",
            description="A custom benchmark",
            category="custom",
            tags=["custom"],
            module_path="custom_pkg.custom",
            function_name="custom_bench",
        ),
        "mmlu": BenchmarkMetadata(  # This should override
            name="Custom MMLU",
            description="A custom version of MMLU",
            category="custom",
            tags=["custom"],
            module_path="custom_pkg.mmlu",
            function_name="custom_mmlu",
        ),
    }

    # Create a mock function that returns the dict
    mock_func = Mock(return_value=benchmarks_dict)

    # Create a mock entry point
    mock_ep = Mock()
    mock_ep.name = "custom_benchmarks"
    mock_ep.load.return_value = mock_func

    # Mock entry_points().select() to return our mock
    mock_select = Mock(return_value=[mock_ep])
    mock_eps = Mock()
    mock_eps.select = mock_select
    mock_entry_points.return_value = mock_eps

    # Load entry points
    result = _load_entry_point_benchmarks()

    # Verify both were loaded
    assert "custom_bench" in result
    assert "mmlu" in result
    assert result["mmlu"].name == "Custom MMLU"
    mock_func.assert_called_once()
