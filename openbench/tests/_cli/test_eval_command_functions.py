"""Simple unit tests for eval command helper functions."""

import pytest
from openbench._cli.eval_command import (
    parse_limit,
    validate_model_name,
    validate_model_role,
)


def test_parse_limit_none():
    """Test parsing None limit."""
    assert parse_limit(None) is None


def test_parse_limit_single():
    """Test parsing single integer limit."""
    assert parse_limit("10") == 10


def test_parse_limit_range():
    """Test parsing range limit."""
    assert parse_limit("5,15") == (5, 15)


def test_parse_limit_invalid():
    """Test invalid limit raises error."""
    with pytest.raises(Exception):
        parse_limit("invalid")


def test_validate_model_name_valid():
    """Test valid model name."""
    # Should not raise
    validate_model_name("provider/model-name")


def test_validate_model_name_invalid():
    """Test invalid model name."""
    with pytest.raises(Exception):
        validate_model_name("invalid-model")


def test_validate_model_role_empty():
    """Test empty model role."""
    assert validate_model_role(None) == {}
    assert validate_model_role("") == {}


def test_validate_model_role_valid():
    """Test valid model role."""
    result = validate_model_role("grader=provider/model")
    assert result == {"grader": "provider/model"}


def test_validate_model_role_invalid():
    """Test invalid model role."""
    with pytest.raises(Exception):
        validate_model_role("invalid-format")
