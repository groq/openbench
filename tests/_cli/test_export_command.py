"""Unit tests for export command."""

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from openbench._cli import app

runner = CliRunner()


def strip_ansi_codes(text: str) -> str:
    """Strip ANSI color codes from text for reliable string matching in tests."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def create_mock_eval_log(
    path: Path, task: str = "gpqa-diamond", model: str = "groq/llama-3.1-70b"
) -> None:
    """Create a mock eval log file for testing.

    Args:
        path: Path to create the log file at
        task: Task name for the eval
        model: Model name for the eval
    """
    log_data = {
        "eval": {
            "eval_id": "test_eval_id",
            "run_id": "test_run_id",
            "created": "2025-01-01T00:00:00",
            "task": task,
            "task_id": "test_task_id",
            "model": model,
        },
        "results": {
            "total_samples": 100,
            "completed_samples": 100,
            "scores": [{"metrics": {"accuracy": {"value": 0.85}}}],
        },
        "stats": {
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T01:00:00",
            "model_usage": {
                "groq/llama-3.1-70b": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                }
            },
        },
        "samples": [
            {
                "id": 1,
                "epoch": 1,
                "target": "A",
                "messages": [{"role": "user", "content": "Test question"}],
                "metadata": {"category": "science"},
                "scores": {"accuracy": {"value": 1.0, "answer": "A"}},
            }
        ],
    }

    if path.suffix == ".json":
        path.write_text(json.dumps(log_data))
    elif path.suffix == ".eval":
        path.write_text(json.dumps(log_data))


class TestExportCommand:
    """Test export command basic functionality."""

    def test_export_requires_log_files(self):
        """Test export command requires log file arguments."""
        result = runner.invoke(app, ["export-hf", "--hub-repo", "test/repo"])
        assert result.exit_code != 0

    def test_export_requires_hub_repo(self):
        """Test export command requires hub-repo option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.eval"
            create_mock_eval_log(log_file)

            result = runner.invoke(app, ["export-hf", "--log-files", str(log_file)])
            assert result.exit_code != 0
            assert "Missing option" in result.stdout

    def test_export_with_nonexistent_file(self):
        """Test export with nonexistent log file."""
        result = runner.invoke(
            app,
            [
                "export-hf",
                "--log-files",
                "/nonexistent/file.eval",
                "--hub-repo",
                "test/repo",
            ],
        )
        assert result.exit_code == 1
        assert "Warning: Log file not found" in result.stdout

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_json_file_success(self, mock_read_log, mock_dataset_class):
        """Test successful export of JSON log file."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "gpqa-diamond",
                "task_id": "task_id",
                "model": "groq/llama-3.1-70b",
            },
            "results": {
                "total_samples": 10,
                "completed_samples": 10,
                "scores": [{"metrics": {"accuracy": {"value": 0.8}}}],
            },
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.return_value = mock_log_data

        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.json"
            log_file.write_text(json.dumps(mock_log_data))

            result = runner.invoke(
                app,
                ["export-hf", "--log-files", str(log_file), "--hub-repo", "test/repo"],
            )

            assert result.exit_code == 0
            assert "Exporting 1 eval log(s) to test/repo" in result.stdout
            assert "✅ Export complete!" in result.stdout

            assert mock_dataset.push_to_hub.call_count >= 1

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_eval_file_success(self, mock_read_log, mock_dataset_class):
        """Test successful export of .eval log file."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "mmlu",
                "task_id": "task_id",
                "model": "openai/gpt-4",
            },
            "results": {
                "total_samples": 5,
                "completed_samples": 5,
                "scores": [{"metrics": {"accuracy": {"value": 0.9}}}],
            },
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.return_value = mock_log_data
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.eval"
            log_file.write_text("dummy")

            result = runner.invoke(
                app,
                [
                    "export-hf",
                    "--log-files",
                    str(log_file),
                    "--hub-repo",
                    "username/openbench-logs",
                ],
            )

            assert result.exit_code == 0
            assert "Exporting 1 eval log(s)" in result.stdout
            assert mock_dataset.push_to_hub.call_count >= 1

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_multiple_files(self, mock_read_log, mock_dataset_class):
        """Test export with multiple log files."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "humaneval",
                "task_id": "task_id",
                "model": "anthropic/claude-3",
            },
            "results": {
                "total_samples": 3,
                "completed_samples": 3,
                "scores": [{"metrics": {"pass@1": {"value": 0.75}}}],
            },
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.return_value = mock_log_data
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file1 = Path(temp_dir) / "test1.json"
            log_file2 = Path(temp_dir) / "test2.json"
            log_file1.write_text(json.dumps(mock_log_data))
            log_file2.write_text(json.dumps(mock_log_data))

            result = runner.invoke(
                app,
                [
                    "export-hf",
                    "--log-files",
                    str(log_file1),
                    "--log-files",
                    str(log_file2),
                    "--hub-repo",
                    "test/repo",
                ],
            )

            assert result.exit_code == 0
            assert "Exporting 2 eval log(s)" in result.stdout

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_with_hub_private(self, mock_read_log, mock_dataset_class):
        """Test export with --hub-private flag."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "test_task",
                "task_id": "task_id",
                "model": "test/model",
            },
            "results": {"total_samples": 1, "completed_samples": 1, "scores": []},
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.return_value = mock_log_data
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.json"
            log_file.write_text(json.dumps(mock_log_data))

            result = runner.invoke(
                app,
                [
                    "export-hf",
                    "--log-files",
                    str(log_file),
                    "--hub-repo",
                    "test/repo",
                    "--hub-private",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_dataset.push_to_hub.call_args_list[0]
            assert call_args[1]["private"] is True

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_with_custom_names(self, mock_read_log, mock_dataset_class):
        """Test export with custom benchmark and model names."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "original_task",
                "task_id": "task_id",
                "model": "original/model",
            },
            "results": {"total_samples": 1, "completed_samples": 1, "scores": []},
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.return_value = mock_log_data
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.json"
            log_file.write_text(json.dumps(mock_log_data))

            result = runner.invoke(
                app,
                [
                    "export-hf",
                    "--log-files",
                    str(log_file),
                    "--hub-repo",
                    "test/repo",
                    "--hub-benchmark-name",
                    "custom-benchmark",
                    "--hub-model-name",
                    "custom-model",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_dataset.push_to_hub.call_args_list[0]
            config_name = call_args[1]["config_name"]
            assert (
                "custom-benchmark" in config_name or "custom_benchmark" in config_name
            )

    @patch("openbench._cli.export_command._read_log_json")
    def test_export_with_malformed_log(self, mock_read_log):
        """Test export handles malformed log files gracefully."""
        mock_read_log.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "malformed.json"
            log_file.write_text("not valid json {{{")

            result = runner.invoke(
                app,
                ["export-hf", "--log-files", str(log_file), "--hub-repo", "test/repo"],
            )

            assert result.exit_code == 1
            assert "Skipping log" in result.stdout
            assert "No valid logs could be parsed" in result.stdout

    @patch("openbench._cli.export_command.Dataset")
    @patch("openbench._cli.export_command._read_log_json")
    def test_export_skips_invalid_continues_with_valid(
        self, mock_read_log, mock_dataset_class
    ):
        """Test export skips invalid logs but continues with valid ones."""
        # First call raises error, second succeeds
        valid_log = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "test",
                "task_id": "task_id",
                "model": "test/model",
            },
            "results": {"total_samples": 1, "completed_samples": 1, "scores": []},
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }
        mock_read_log.side_effect = [json.JSONDecodeError("Error", "", 0), valid_log]
        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file1 = Path(temp_dir) / "invalid.json"
            log_file2 = Path(temp_dir) / "valid.json"
            log_file1.write_text("invalid")
            log_file2.write_text(json.dumps(valid_log))

            result = runner.invoke(
                app,
                [
                    "export-hf",
                    "--log-files",
                    str(log_file1),
                    "--log-files",
                    str(log_file2),
                    "--hub-repo",
                    "test/repo",
                ],
            )

            assert result.exit_code == 0
            assert "Skipping log" in result.stdout
            assert "✅ Export complete!" in result.stdout

    def test_export_help(self):
        """Test export help command."""
        result = runner.invoke(app, ["export-hf", "--help"])
        assert result.exit_code == 0
        clean_stdout = strip_ansi_codes(result.stdout)
        assert "Export evaluation log files to HuggingFace Hub" in clean_stdout
        assert "--hub-repo" in clean_stdout
        assert "--hub-private" in clean_stdout
        assert "--hub-benchmark-name" in clean_stdout
        assert "--hub-model-name" in clean_stdout


class TestExportCommandIntegration:
    """Integration tests for export command with actual file operations."""

    @patch("openbench._cli.export_command.Dataset")
    @patch("subprocess.run")
    def test_export_eval_file_uses_inspect_dump(
        self, mock_subprocess, mock_dataset_class
    ):
        """Test that .eval files use inspect log dump command."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "test",
                "task_id": "task_id",
                "model": "test/model",
            },
            "results": {"total_samples": 1, "completed_samples": 1, "scores": []},
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {},
            },
            "samples": [],
        }

        mock_process = MagicMock()
        mock_process.stdout = json.dumps(mock_log_data)
        mock_subprocess.return_value = mock_process

        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.eval"
            log_file.write_text("binary eval data")

            result = runner.invoke(
                app,
                ["export-hf", "--log-files", str(log_file), "--hub-repo", "test/repo"],
            )

            assert result.exit_code == 0
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "inspect" in call_args
            assert "log" in call_args
            assert "dump" in call_args

    @patch("openbench._cli.export_command.Dataset")
    def test_export_creates_three_datasets(self, mock_dataset_class):
        """Test that export creates results, stats, and samples datasets."""
        mock_log_data = {
            "eval": {
                "eval_id": "test_id",
                "run_id": "run_id",
                "created": "2025-01-01T00:00:00",
                "task": "test",
                "task_id": "task_id",
                "model": "test/model",
            },
            "results": {
                "total_samples": 1,
                "completed_samples": 1,
                "scores": [{"metrics": {"accuracy": {"value": 0.5}}}],
            },
            "stats": {
                "started_at": "2025-01-01T00:00:00",
                "completed_at": "2025-01-01T01:00:00",
                "model_usage": {
                    "test/model": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    }
                },
            },
            "samples": [
                {"id": 1, "epoch": 1, "target": "A", "messages": [], "scores": {}}
            ],
        }

        mock_dataset = MagicMock()
        mock_dataset_class.from_list.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.json"
            log_file.write_text(json.dumps(mock_log_data))

            result = runner.invoke(
                app,
                ["export-hf", "--log-files", str(log_file), "--hub-repo", "test/repo"],
            )

            assert result.exit_code == 0

            # Should create 3 datasets: results, stats, samples
            assert mock_dataset_class.from_list.call_count == 3
            assert mock_dataset.push_to_hub.call_count == 3

            # Check config names
            config_names = [
                call[1]["config_name"]
                for call in mock_dataset.push_to_hub.call_args_list
            ]
            assert any("results" in name for name in config_names)
            assert any("stats" in name for name in config_names)
            assert any("samples" in name for name in config_names)
