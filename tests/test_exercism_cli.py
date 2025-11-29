"""Tests for Exercism-specific CLI helpers."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from openbench.utils import cli_commands


class _RecordingSandbox:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
        self.commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class _LocalSandbox:
    async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        return SimpleNamespace(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@pytest.mark.asyncio
async def test_discover_hidden_paths_detects_spec_js(tmp_path, monkeypatch):
    """Ensure .spec.js files are treated as hidden test files."""
    task_dir = tmp_path / "javascript" / "two-fer"
    (task_dir / "src").mkdir(parents=True)
    (task_dir / "src" / "two_fer.spec.js").write_text("// spec file")

    local_sandbox = _LocalSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: local_sandbox)

    result = await cli_commands.discover_hidden_paths(str(task_dir))

    assert result["success"] is True
    hidden_files = result["hidden_paths"]["files"]
    assert "src/two_fer.spec.js" in hidden_files


@pytest.mark.asyncio
async def test_prepare_hidden_workspace_builds_excludes(monkeypatch):
    """prepare_hidden_workspace should exclude discovered test artifacts."""

    async def fake_discover(path: str):
        assert path == "/workspace/javascript/two-fer"
        return {
            "success": True,
            "hidden_paths": {"files": ["src/two_fer.spec.js"], "dirs": []},
        }

    recording_sandbox = _RecordingSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: recording_sandbox)
    monkeypatch.setattr(cli_commands, "discover_hidden_paths", fake_discover)

    result = await cli_commands.prepare_hidden_workspace("javascript", "two-fer")

    assert result["success"] is True
    assert result["agent_dir"].endswith("/javascript/two-fer")
    # Verify rsync command contains exclude flag for the spec file
    rsync_script = recording_sandbox.commands[-1][2]
    assert "--exclude=src/two_fer.spec.js" in rsync_script


@pytest.mark.asyncio
async def test_sync_agent_workspace_respects_hidden_paths(monkeypatch):
    """sync_agent_workspace should keep hidden paths excluded during sync."""
    recording_sandbox = _RecordingSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: recording_sandbox)

    hidden_paths = {"files": ["src/two_fer.spec.js"], "dirs": ["testdata"]}

    result = await cli_commands.sync_agent_workspace(
        "/tmp/agent", "/tmp/full", hidden_paths
    )

    assert result["success"] is True
    script = recording_sandbox.commands[-1][2]
    assert "--exclude=src/two_fer.spec.js" in script
    assert "--exclude=testdata" in script
    assert "--exclude='testdata/**'" in script


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================


def test_normalize_relative_path_edge_cases():
    """Test path normalization edge cases."""
    from openbench.utils.cli_commands import _normalize_relative_path

    # Empty string
    assert _normalize_relative_path("") == ""

    # Current directory
    assert _normalize_relative_path(".") == ""
    assert _normalize_relative_path("./") == ""

    # Leading ./ should be stripped
    assert _normalize_relative_path("./src/file.py") == "src/file.py"

    # Nested paths with ..
    assert _normalize_relative_path("./foo/../bar") == "bar"

    # Already normalized paths
    assert _normalize_relative_path("src/test/file.py") == "src/test/file.py"

    # Trailing slashes
    assert _normalize_relative_path("./src/") == "src"


def test_build_exclude_flags_empty_inputs():
    """Test exclude flag building with empty inputs."""
    from openbench.utils.cli_commands import _build_exclude_flags

    # Empty dict
    assert _build_exclude_flags({}) == []

    # Empty files and dirs
    assert _build_exclude_flags({"files": [], "dirs": []}) == []


def test_build_exclude_flags_files_only():
    """Test exclude flags with only files."""
    from openbench.utils.cli_commands import _build_exclude_flags

    flags = _build_exclude_flags({"files": ["test_file.py", "src/test.js"], "dirs": []})

    assert len(flags) == 2
    assert "--exclude=test_file.py" in flags
    assert "--exclude=src/test.js" in flags


def test_build_exclude_flags_dirs_only():
    """Test exclude flags with only directories."""
    from openbench.utils.cli_commands import _build_exclude_flags

    flags = _build_exclude_flags({"files": [], "dirs": ["tests", "src/test_data"]})

    # Each dir should have 2 flags (dir itself and dir/**)
    assert len(flags) == 4
    assert "--exclude=tests" in flags
    assert "--exclude='tests/**'" in flags


def test_build_exclude_flags_skips_empty_paths():
    """Test that empty paths are skipped."""
    from openbench.utils.cli_commands import _build_exclude_flags

    flags = _build_exclude_flags(
        {"files": ["", "valid.py", "."], "dirs": ["", ".", "valid_dir"]}
    )

    # Should only include valid paths
    assert "--exclude=valid.py" in flags
    assert "--exclude=valid_dir" in flags


# =============================================================================
# Additional discover_hidden_paths Tests
# =============================================================================


@pytest.mark.asyncio
async def test_discover_hidden_paths_detects_test_dirs(tmp_path, monkeypatch):
    """Ensure directories with 'test' in name are detected."""
    task_dir = tmp_path / "python" / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "src").mkdir(parents=True)
    (task_dir / "src" / "test_utils").mkdir(parents=True)

    local_sandbox = _LocalSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: local_sandbox)

    result = await cli_commands.discover_hidden_paths(str(task_dir))

    assert result["success"] is True
    hidden_dirs = result["hidden_paths"]["dirs"]
    assert "tests" in hidden_dirs
    assert "src/test_utils" in hidden_dirs


@pytest.mark.asyncio
async def test_discover_hidden_paths_case_insensitive(tmp_path, monkeypatch):
    """Ensure detection is case-insensitive."""
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "TEST_file.py").write_text("")
    (task_dir / "Test_Dir").mkdir()

    local_sandbox = _LocalSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: local_sandbox)

    result = await cli_commands.discover_hidden_paths(str(task_dir))

    assert result["success"] is True
    assert "TEST_file.py" in result["hidden_paths"]["files"]
    assert "Test_Dir" in result["hidden_paths"]["dirs"]


@pytest.mark.asyncio
async def test_discover_hidden_paths_handles_script_failure(monkeypatch):
    """Test error handling when script fails."""

    class FailingSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            return SimpleNamespace(returncode=1, stdout="", stderr="Python error")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: FailingSandbox())

    result = await cli_commands.discover_hidden_paths("/some/path")

    assert result["success"] is False
    assert "stderr" in result
    assert "Python error" in result["stderr"]


@pytest.mark.asyncio
async def test_discover_hidden_paths_handles_invalid_json(monkeypatch):
    """Test error handling when JSON parsing fails."""

    class BadJsonSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            return SimpleNamespace(returncode=0, stdout="not valid json{", stderr="")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: BadJsonSandbox())

    result = await cli_commands.discover_hidden_paths("/some/path")

    assert result["success"] is False
    assert "failed to parse" in result["stderr"]


@pytest.mark.asyncio
async def test_discover_hidden_paths_no_test_files(tmp_path, monkeypatch):
    """Test with directory containing no test files."""
    task_dir = tmp_path / "clean_task"
    task_dir.mkdir(parents=True)
    (task_dir / "main.py").write_text("")
    (task_dir / "utils.py").write_text("")

    local_sandbox = _LocalSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: local_sandbox)

    result = await cli_commands.discover_hidden_paths(str(task_dir))

    assert result["success"] is True
    assert result["hidden_paths"]["files"] == []
    assert result["hidden_paths"]["dirs"] == []


# =============================================================================
# prepare_hidden_workspace Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_prepare_hidden_workspace_propagates_discovery_failure(monkeypatch):
    """Test that discovery failures are propagated."""

    async def failing_discover(path: str):
        return {"success": False, "stdout": "", "stderr": "Discovery failed"}

    monkeypatch.setattr(cli_commands, "discover_hidden_paths", failing_discover)
    monkeypatch.setattr(cli_commands, "sandbox", lambda: _RecordingSandbox())

    result = await cli_commands.prepare_hidden_workspace("python", "task")

    assert result["success"] is False
    assert result["stderr"] == "Discovery failed"


@pytest.mark.asyncio
async def test_prepare_hidden_workspace_no_excludes(monkeypatch):
    """Test workspace preparation when no test files found."""

    async def clean_discover(path: str):
        return {"success": True, "hidden_paths": {"files": [], "dirs": []}}

    recording_sandbox = _RecordingSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: recording_sandbox)
    monkeypatch.setattr(cli_commands, "discover_hidden_paths", clean_discover)

    result = await cli_commands.prepare_hidden_workspace("python", "task")

    assert result["success"] is True
    # Verify rsync command doesn't have --exclude flags
    rsync_script = recording_sandbox.commands[-1][2]
    assert "--exclude" not in rsync_script


# =============================================================================
# run_setup_commands Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_setup_commands_empty_list(monkeypatch):
    """Test with no setup commands."""
    result = await cli_commands.run_setup_commands([], "/workspace/task")
    assert result == "No setup commands"


@pytest.mark.asyncio
async def test_run_setup_commands_success(monkeypatch):
    """Test successful setup command execution."""

    class SuccessSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            return SimpleNamespace(returncode=0, stdout="Setup complete", stderr="")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: SuccessSandbox())

    result = await cli_commands.run_setup_commands(
        ["pip install -r requirements.txt"], "/workspace/task"
    )

    assert "Exit Code: 0" in result
    assert "Success: True" in result
    assert "Setup complete" in result


@pytest.mark.asyncio
async def test_run_setup_commands_failure(monkeypatch):
    """Test setup command failure."""

    class FailingSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            return SimpleNamespace(returncode=1, stdout="", stderr="Package not found")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: FailingSandbox())

    result = await cli_commands.run_setup_commands(
        ["pip install nonexistent"], "/workspace/task"
    )

    assert "Exit Code: 1" in result
    assert "Success: False" in result
    assert "Package not found" in result


# =============================================================================
# run_final_test Tests
# =============================================================================


@pytest.mark.asyncio
async def test_run_final_test_python_filename_normalization(monkeypatch):
    """Test Python test filename normalization (hyphens to underscores)."""

    class RecordingCommandSandbox:
        def __init__(self):
            self.last_cmd = None

        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            self.last_cmd = cmd
            return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    sandbox = RecordingCommandSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: sandbox)

    await cli_commands.run_final_test(
        "pytest two-fer-test.py", "/workspace/python/two-fer"
    )

    # Verify hyphens were converted to underscores
    assert "two_fer_test.py" in sandbox.last_cmd[2]


@pytest.mark.asyncio
async def test_run_final_test_non_python_unchanged(monkeypatch):
    """Test that non-Python tests don't get filename changes."""

    class RecordingCommandSandbox:
        def __init__(self):
            self.last_cmd = None

        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            self.last_cmd = cmd
            return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    sandbox = RecordingCommandSandbox()
    monkeypatch.setattr(cli_commands, "sandbox", lambda: sandbox)

    await cli_commands.run_final_test(
        "npm test two-fer-test.js", "/workspace/javascript/two-fer"
    )

    # Verify filename was NOT changed (no python in path)
    assert "two-fer-test.js" in sandbox.last_cmd[2]


@pytest.mark.asyncio
async def test_run_final_test_exception_handling(monkeypatch):
    """Test exception handling in test execution."""

    class ExceptionSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            raise RuntimeError("Sandbox crashed")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: ExceptionSandbox())

    result = await cli_commands.run_final_test("pytest test.py", "/workspace/task")

    assert "ERROR: test run failed" in result
    assert "Sandbox crashed" in result


# =============================================================================
# ensure_repo_and_task Tests
# =============================================================================


@pytest.mark.asyncio
async def test_ensure_repo_and_task_success(monkeypatch):
    """Test successful repo and task verification."""

    class SuccessSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: SuccessSandbox())

    result = await cli_commands.ensure_repo_and_task("python", "two-fer")
    assert result is True


@pytest.mark.asyncio
async def test_ensure_repo_and_task_missing_task(monkeypatch):
    """Test with missing task directory."""

    class FailingSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            # Fail on the task existence check
            return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: FailingSandbox())

    result = await cli_commands.ensure_repo_and_task("python", "nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_ensure_repo_and_task_exception(monkeypatch):
    """Test exception handling."""

    class ExceptionSandbox:
        async def exec(self, cmd, timeout=0, env=None):  # type: ignore[override]
            raise Exception("Connection failed")

    monkeypatch.setattr(cli_commands, "sandbox", lambda: ExceptionSandbox())

    result = await cli_commands.ensure_repo_and_task("python", "task")
    assert result is False
