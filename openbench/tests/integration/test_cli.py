"""Simple integration tests for the bench CLI tool."""

import os
import pytest
from typer.testing import CliRunner
from openbench._cli import app

runner = CliRunner()

# Mark all tests in this module as integration tests
# Skip all tests if GROQ_API_KEY is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set - skipping integration tests",
    ),
]


def test_help():
    """Test help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.stdout


def test_basic_mmlu():
    """Test basic MMLU evaluation."""
    result = runner.invoke(
        app, ["eval", "mmlu", "--limit", "1", "--model", "groq/llama-3.1-8b-instant"]
    )
    assert result.exit_code == 0


def test_basic_gpqa():
    """Test basic GPQA evaluation."""
    result = runner.invoke(
        app,
        [
            "eval",
            "gpqa_diamond",
            "--limit",
            "1",
            "--model",
            "groq/llama-3.1-8b-instant",
        ],
    )
    assert result.exit_code == 0


def test_basic_humaneval():
    """Test basic HumanEval evaluation."""
    result = runner.invoke(
        app,
        [
            "eval",
            "humaneval",
            "--limit",
            "1",
            "--model",
            "groq/llama-3.1-8b-instant",
            "--epochs",
            "5",
        ],
    )
    assert result.exit_code == 0


def test_invalid_benchmark():
    """Test invalid benchmark name."""
    result = runner.invoke(
        app, ["eval", "invalid_benchmark", "--model", "groq/llama-3.1-8b-instant"]
    )
    assert result.exit_code != 0


def test_invalid_model_format():
    """Test invalid model format."""
    result = runner.invoke(app, ["eval", "mmlu", "--model", "invalid-model"])
    assert result.exit_code != 0


def test_model_and_model_role_conflict():
    """Test conflicting model specifications."""
    result = runner.invoke(
        app,
        [
            "eval",
            "mmlu",
            "--model",
            "groq/llama-3.1-8b-instant",
            "--model-role",
            "candidate=groq/llama-3.1-8b-instant",
        ],
    )
    assert result.exit_code != 0
