"""Tests for eval group configuration."""

import pytest
from openbench.config import EVAL_GROUPS, BENCHMARKS


def test_eval_groups_exist():
    """Test that EVAL_GROUPS is defined and not empty."""
    assert EVAL_GROUPS is not None
    assert len(EVAL_GROUPS) > 0


def test_no_group_benchmark_conflicts():
    """Test that eval group names don't conflict with benchmark names.

    Eval groups should only exist as groups, not as actual benchmark entry points.
    This prevents confusion where a name could refer to either a group or a task.
    """
    group_names = set(EVAL_GROUPS.keys())
    benchmark_names = set(BENCHMARKS.keys())

    conflicts = group_names & benchmark_names
    assert len(conflicts) == 0, (
        f"These names exist as both eval groups and benchmarks: {conflicts}. "
        f"Eval groups should not have corresponding benchmark entry points."
    )


def test_group_benchmarks_are_valid():
    """Test that all benchmarks referenced in groups exist."""
    for group_name, group in EVAL_GROUPS.items():
        for benchmark in group.benchmarks:
            assert benchmark in BENCHMARKS, (
                f"Group '{group_name}' references unknown benchmark '{benchmark}'"
            )
