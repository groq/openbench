"""Unit tests for cache command."""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from openbench._cli import app
from openbench._cli.cache_command import (
    _cache_root,
    _discover_cache_types,
    _get_cache_path,
    _get_default_cache_type,
    _human_size,
    _dir_size,
)

runner = CliRunner()


def strip_ansi_codes(text: str) -> str:
    """Strip ANSI color codes from text for reliable string matching in tests."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class TestCacheHelperFunctions:
    """Test helper functions used by cache commands."""

    def test_cache_root_returns_correct_path(self):
        """Test that _cache_root returns the expected cache root directory path."""
        expected = Path(os.path.expanduser("~/.openbench")).resolve()
        assert _cache_root() == expected

    def test_get_cache_path_with_type(self):
        """Test _get_cache_path with specific cache type."""
        result = _get_cache_path("livemcpbench")
        expected = Path(os.path.expanduser("~/.openbench/livemcpbench")).resolve()
        assert result == expected

    def test_get_cache_path_without_type(self):
        """Test _get_cache_path without cache type returns root."""
        result = _get_cache_path()
        expected = Path(os.path.expanduser("~/.openbench")).resolve()
        assert result == expected

    @patch("openbench._cli.cache_command._cache_root")
    def test_discover_cache_types_empty(self, mock_cache_root):
        """Test _discover_cache_types with no cache directory."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_cache_root.return_value = mock_path

        result = _discover_cache_types()
        assert result == []

    @patch("openbench._cli.cache_command._cache_root")
    def test_discover_cache_types_with_caches(self, mock_cache_root):
        """Test _discover_cache_types with multiple cache types."""
        mock_dir1 = MagicMock()
        mock_dir1.name = "livemcpbench"
        mock_dir1.is_dir.return_value = True

        mock_dir2 = MagicMock()
        mock_dir2.name = "scicode"
        mock_dir2.is_dir.return_value = True

        mock_file = MagicMock()
        mock_file.name = "somefile.txt"
        mock_file.is_dir.return_value = False

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.iterdir.return_value = [mock_dir1, mock_dir2, mock_file]
        mock_cache_root.return_value = mock_path

        result = _discover_cache_types()
        assert set(result) == {"livemcpbench", "scicode"}

    @patch("openbench._cli.cache_command._discover_cache_types")
    def test_get_default_cache_type_single_livemcp(self, mock_discover):
        """Test _get_default_cache_type returns livemcpbench when it's the only type."""
        mock_discover.return_value = ["livemcpbench"]
        result = _get_default_cache_type()
        assert result == "livemcpbench"

    @patch("openbench._cli.cache_command._discover_cache_types")
    def test_get_default_cache_type_multiple_types(self, mock_discover):
        """Test _get_default_cache_type returns None when multiple types exist."""
        mock_discover.return_value = ["livemcpbench", "scicode"]
        result = _get_default_cache_type()
        assert result is None

    @patch("openbench._cli.cache_command._discover_cache_types")
    def test_get_default_cache_type_no_types(self, mock_discover):
        """Test _get_default_cache_type returns None when no types exist."""
        mock_discover.return_value = []
        result = _get_default_cache_type()
        assert result is None

    def test_human_size_formats_bytes_correctly(self):
        """Test human-readable size formatting."""
        assert _human_size(0) == "0.0 B"
        assert _human_size(512) == "512.0 B"
        assert _human_size(1024) == "1.0 KB"
        assert _human_size(1536) == "1.5 KB"
        assert _human_size(1024 * 1024) == "1.0 MB"
        assert _human_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _human_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"
        assert _human_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.0 PB"

    def test_dir_size_nonexistent_directory(self):
        """Test _dir_size returns 0 for nonexistent directory."""
        nonexistent_path = Path("/nonexistent/directory")
        assert _dir_size(nonexistent_path) == 0

    def test_dir_size_empty_directory(self):
        """Test _dir_size returns 0 for empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert _dir_size(Path(temp_dir)) == 0

    def test_dir_size_single_file(self):
        """Test _dir_size returns correct size for a single file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"hello world")
            temp_file.flush()

            try:
                assert _dir_size(Path(temp_file.name)) == 11
            finally:
                os.unlink(temp_file.name)

    def test_dir_size_directory_with_files(self):
        """Test _dir_size calculates total size of directory with files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with known sizes
            (temp_path / "file1.txt").write_bytes(b"a" * 100)
            (temp_path / "file2.txt").write_bytes(b"b" * 200)

            # Create subdirectory with file
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "file3.txt").write_bytes(b"c" * 50)

            assert _dir_size(temp_path) == 350

    def test_dir_size_handles_permission_errors(self):
        """Test _dir_size handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "file.txt").write_bytes(b"test")

            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.side_effect = PermissionError("Access denied")

            # Mock Path.rglob at the module level
            with patch("pathlib.Path.rglob", return_value=[mock_file]):
                result = _dir_size(temp_path)
                assert isinstance(result, int)
                assert result == 0


class TestCacheInfoCommand:
    """Test cache info command."""

    @patch("openbench._cli.cache_command._discover_cache_types")
    def test_cache_info_no_cache_directory(self, mock_discover):
        """Test cache info when no cache directories exist."""
        mock_discover.return_value = []

        result = runner.invoke(app, ["cache", "info"])
        assert result.exit_code == 0
        assert "No cache directories found." in result.stdout

    @patch("openbench._cli.cache_command._dir_size")
    @patch("openbench._cli.cache_command._discover_cache_types")
    @patch("openbench._cli.cache_command._cache_root")
    def test_cache_info_with_cache_types(
        self, mock_cache_root, mock_discover, mock_dir_size
    ):
        """Test cache info with multiple cache types."""
        mock_root = MagicMock()
        mock_cache_root.return_value = mock_root
        mock_discover.return_value = ["livemcpbench", "scicode"]
        mock_dir_size.side_effect = [1024, 512, 256]  # total, livemcpbench, scicode

        result = runner.invoke(app, ["cache", "info"])
        assert result.exit_code == 0
        assert "Total size: 1.0 KB" in result.stdout
        assert "livemcpbench" in result.stdout
        assert "scicode" in result.stdout

    @patch("openbench._cli.cache_command._dir_size")
    @patch("openbench._cli.cache_command._get_cache_path")
    def test_cache_info_specific_type(self, mock_get_cache_path, mock_dir_size):
        """Test cache info with specific cache type."""
        # Mock subdirectories
        subdir1 = MagicMock()
        subdir1.name = "subdir1"
        subdir1.is_dir.return_value = True

        subdir2 = MagicMock()
        subdir2.name = "subdir2"
        subdir2.is_dir.return_value = True

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.iterdir.return_value = [subdir1, subdir2]
        mock_get_cache_path.return_value = mock_path

        # Mock sizes
        mock_dir_size.side_effect = [1024, 512, 256]  # total, subdir1, subdir2

        result = runner.invoke(app, ["cache", "info", "--type", "livemcpbench"])
        assert result.exit_code == 0
        assert "Total size: 1.0 KB" in result.stdout
        assert "subdir1" in result.stdout
        assert "subdir2" in result.stdout

    def test_cache_info_shows_subdirectories(self):
        """Test cache info shows subdirectories for each cache type."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / ".openbench"

            # Create livemcpbench cache with subdirectories
            lmcp_dir = cache_root / "livemcpbench"
            lmcp_dir.mkdir(parents=True)
            (lmcp_dir / "copilot").mkdir()
            (lmcp_dir / "root").mkdir()
            (lmcp_dir / "copilot" / "test.txt").write_text("test data")
            (lmcp_dir / "root" / "data.json").write_text("{}")

            # Create scicode cache with subdirectories
            sci_dir = cache_root / "scicode"
            sci_dir.mkdir()
            (sci_dir / "downloads").mkdir()
            (sci_dir / "downloads" / "file.h5").write_text("binary data")

            # Mock the cache root to point to our temp directory
            with patch(
                "openbench._cli.cache_command._cache_root", return_value=cache_root
            ):
                result = runner.invoke(app, ["cache", "info"])
                assert result.exit_code == 0

                # Check that cache types are shown
                assert "livemcpbench" in result.stdout
                assert "scicode" in result.stdout

                # Check that subdirectories are shown with tree symbols
                assert "└─ copilot" in result.stdout
                assert "└─ root" in result.stdout
                assert "└─ downloads" in result.stdout


class TestCacheLsCommand:
    """Test cache ls command."""

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_nonexistent_path(self, mock_default_type, mock_get_cache_path):
        """Test cache ls with nonexistent path."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "ls"])
        assert result.exit_code == 1
        assert "Path not found:" in result.stdout

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_empty_directory(self, mock_default_type, mock_get_cache_path):
        """Test cache ls with empty directory."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.iterdir.return_value = []
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "ls"])
        assert result.exit_code == 0
        assert "(empty)" in result.stdout

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_with_files(self, mock_default_type, mock_get_cache_path):
        """Test cache ls with files and directories."""
        mock_default_type.return_value = "livemcpbench"
        # Mock file and directory
        mock_file = MagicMock()
        mock_file.name = "file.txt"
        mock_file.is_dir.return_value = False

        mock_dir = MagicMock()
        mock_dir.name = "subdir"
        mock_dir.is_dir.return_value = True

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.iterdir.return_value = [mock_file, mock_dir]
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "ls"])
        assert result.exit_code == 0
        assert "subdir/" in result.stdout
        assert "file.txt" in result.stdout

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_single_file(self, mock_default_type, mock_get_cache_path):
        """Test cache ls when target is a single file."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "ls"])
        assert result.exit_code == 0
        assert "(file)" in result.stdout

    @patch("openbench._cli.cache_command._print_tree")
    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_tree_view(
        self, mock_default_type, mock_get_cache_path, mock_print_tree
    ):
        """Test cache ls with tree view."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "ls", "--tree"])
        assert result.exit_code == 0
        mock_print_tree.assert_called_once_with(mock_path)

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_ls_with_subpath(self, mock_default_type, mock_get_cache_path):
        """Test cache ls with subpath option."""
        mock_default_type.return_value = "livemcpbench"
        mock_base = MagicMock()
        mock_subpath = MagicMock()
        mock_subpath.exists.return_value = True
        mock_subpath.is_dir.return_value = True
        mock_subpath.iterdir.return_value = []

        mock_base.__truediv__.return_value = mock_subpath
        mock_get_cache_path.return_value = mock_base

        result = runner.invoke(app, ["cache", "ls", "--path", "subdir"])
        assert result.exit_code == 0
        assert "(empty)" in result.stdout


class TestCacheClearCommand:
    """Test cache clear command."""

    def test_cache_clear_missing_arguments(self):
        """Test cache clear without required arguments."""
        result = runner.invoke(app, ["cache", "clear"])
        assert result.exit_code == 2
        assert (
            "Specify --all to clear everything or --path to clear a subpath."
            in result.stdout
        )

    def test_cache_clear_conflicting_arguments(self):
        """Test cache clear with conflicting arguments."""
        result = runner.invoke(app, ["cache", "clear", "--all", "--path", "subdir"])
        assert result.exit_code == 2
        assert "Specify either --all or --path, not both." in result.stdout

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_nonexistent_path(self, mock_default_type, mock_get_cache_path):
        """Test cache clear with nonexistent path."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
        assert result.exit_code == 1
        assert "Path not found:" in result.stdout

    @patch("typer.confirm")
    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_user_aborts(
        self, mock_default_type, mock_get_cache_path, mock_confirm
    ):
        """Test cache clear when user aborts confirmation."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_get_cache_path.return_value = mock_path
        mock_confirm.return_value = False

        result = runner.invoke(app, ["cache", "clear", "--all"])
        assert result.exit_code == 0
        assert "Aborted." in result.stdout

    @patch("shutil.rmtree")
    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_directory_success(
        self, mock_default_type, mock_get_cache_path, mock_rmtree
    ):
        """Test successful cache clear of directory."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
        assert result.exit_code == 0
        assert "✅ Cleared." in result.stdout
        mock_rmtree.assert_called_once_with(mock_path)

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_file_success(self, mock_default_type, mock_get_cache_path):
        """Test successful cache clear of single file."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_get_cache_path.return_value = mock_path

        result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
        assert result.exit_code == 0
        assert "✅ Cleared." in result.stdout
        mock_path.unlink.assert_called_once()

    @patch("shutil.rmtree")
    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_failure(
        self, mock_default_type, mock_get_cache_path, mock_rmtree
    ):
        """Test cache clear failure handling."""
        mock_default_type.return_value = "livemcpbench"
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        mock_get_cache_path.return_value = mock_path
        mock_rmtree.side_effect = PermissionError("Access denied")

        result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
        assert result.exit_code == 1
        assert "❌ Failed to clear:" in result.stdout
        assert "Access denied" in result.stdout

    @patch("openbench._cli.cache_command._get_cache_path")
    @patch("openbench._cli.cache_command._get_default_cache_type")
    def test_cache_clear_with_subpath(self, mock_default_type, mock_get_cache_path):
        """Test cache clear with specific subpath."""
        mock_default_type.return_value = "livemcpbench"
        mock_base = MagicMock()
        mock_subpath = MagicMock()
        mock_subpath.exists.return_value = True
        mock_subpath.is_file.return_value = True

        mock_base.__truediv__.return_value = mock_subpath
        mock_get_cache_path.return_value = mock_base

        result = runner.invoke(app, ["cache", "clear", "--path", "subdir", "--yes"])
        assert result.exit_code == 0
        assert "✅ Cleared." in result.stdout
        mock_subpath.unlink.assert_called_once()


class TestCacheCommandIntegration:
    """Integration tests for cache command."""

    def test_cache_help(self):
        """Test cache help command."""
        result = runner.invoke(app, ["cache", "--help"])
        assert result.exit_code == 0
        clean_stdout = strip_ansi_codes(result.stdout)
        assert "Manage OpenBench caches" in clean_stdout
        assert "info" in clean_stdout
        assert "ls" in clean_stdout
        assert "clear" in clean_stdout

    def test_cache_info_help(self):
        """Test cache info help command."""
        result = runner.invoke(app, ["cache", "info", "--help"])
        assert result.exit_code == 0
        clean_stdout = strip_ansi_codes(result.stdout)
        assert "Show cache information and sizes" in clean_stdout

    def test_cache_ls_help(self):
        """Test cache ls help command."""
        result = runner.invoke(app, ["cache", "ls", "--help"])
        assert result.exit_code == 0
        clean_stdout = strip_ansi_codes(result.stdout)
        assert "List cache contents" in clean_stdout
        assert "--path" in clean_stdout
        assert "--tree" in clean_stdout

    def test_cache_clear_help(self):
        """Test cache clear help command."""
        result = runner.invoke(app, ["cache", "clear", "--help"])
        assert result.exit_code == 0
        clean_stdout = strip_ansi_codes(result.stdout)
        assert "Remove selected cache data" in clean_stdout
        assert "--all" in clean_stdout
        assert "--path" in clean_stdout
        assert "--yes" in clean_stdout


class TestPrintTreeFunction:
    """Test the _print_tree helper function."""

    @patch("typer.echo")
    def test_print_tree_called_recursively(self, mock_echo):
        """Test that _print_tree is called recursively for directories."""
        # Import the actual function to test it
        from openbench._cli.cache_command import _print_tree

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a simple directory structure
            (temp_path / "file.txt").write_text("test")
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested_file.txt").write_text("nested")

            # Call the actual function
            _print_tree(temp_path)

            # Verify typer.echo was called (indicating output was generated)
            assert mock_echo.call_count > 0

    @patch("typer.echo")
    def test_print_tree_nonexistent_path(self, mock_echo):
        """Test _print_tree with nonexistent path."""
        from openbench._cli.cache_command import _print_tree

        nonexistent_path = Path("/nonexistent/path")
        _print_tree(nonexistent_path)

        # Should print error message
        mock_echo.assert_called_with(f"Path not found: {nonexistent_path}")
