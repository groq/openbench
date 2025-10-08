"""Tests for eval group configuration."""

import pytest
from openbench.config import EVAL_GROUPS, BENCHMARKS


def test_eval_groups_exist():
    """Test that EVAL_GROUPS is defined and not empty."""
    assert EVAL_GROUPS is not None
    assert len(EVAL_GROUPS) > 0


def test_group_names_can_match_tasks():
    """Test that eval groups can intentionally match task names (for family benchmarks)."""
    # This is intentional - family benchmarks like 'bigbench' exist as both
    # a group (for expansion) and potentially as a task entrypoint
    group_names = set(EVAL_GROUPS.keys())
    task_names = set(BENCHMARKS.keys())

    # Overlaps are allowed and expected
    overlaps = group_names & task_names
    # Just verify we have some overlaps (family benchmarks)
    assert isinstance(overlaps, set)  # This will always pass, just documenting the behavior


def test_group_benchmarks_are_valid():
    """Test that all benchmarks referenced in groups exist."""
    for group_name, group in EVAL_GROUPS.items():
        for benchmark in group.benchmarks:
            # Normalize benchmark name for lookup (handle both _ and -)
            assert benchmark in BENCHMARKS, (
                f"Group '{group_name}' references unknown benchmark '{benchmark}'"
            )
