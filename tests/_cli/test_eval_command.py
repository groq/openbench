"""Simple unit tests for eval command."""

from typer.testing import CliRunner
from openbench._cli import app

runner = CliRunner()


def test_eval_requires_benchmark():
    """Test eval command requires a benchmark argument."""
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0


def test_invalid_limit():
    """Test invalid limit parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--limit", "invalid"])
    assert result.exit_code != 0


def test_invalid_display():
    """Test invalid display parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--display", "invalid"])
    assert result.exit_code != 0


def test_invalid_sandbox():
    """Test invalid sandbox parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--sandbox", "invalid"])
    assert result.exit_code != 0


def test_invalid_reasoning_effort():
    """Test invalid reasoning effort parameter."""
    result = runner.invoke(app, ["eval", "mmlu", "--reasoning-effort", "invalid"])
    assert result.exit_code != 0


def test_system_prompt_accepted():
    """Test that system prompt parameter is accepted."""
    # This test doesn't actually run the eval, but verifies the CLI accepts the parameter
    # We expect it to fail on model/benchmark validation, not on the system-prompt parameter
    result = runner.invoke(
        app, ["eval", "mmlu", "--system-prompt", "Think step by step."]
    )
    # The command should fail on trying to run the actual eval, not on parsing the system-prompt
    # So we check that the error is not about an unknown option
    assert "--system-prompt" not in result.output or "no such option" not in result.output.lower()
