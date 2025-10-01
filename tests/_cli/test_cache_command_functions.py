"""Unit tests for cache command helper functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from openbench._cli.cache_command import (
    _cache_root,
    _discover_cache_types,
    _get_cache_path,
    _get_default_cache_type,
    _human_size,
    _dir_size,
    _print_tree,
)


class TestCacheRoot:
    """Test _cache_root function."""

    def test_cache_root_returns_resolved_path(self):
        """Test that _cache_root returns a resolved Path object."""
        result = _cache_root()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert str(result).endswith(".openbench")

    def test_cache_root_expands_user_home(self):
        """Test that _cache_root properly expands ~ to user home."""
        result = _cache_root()
        assert "~" not in str(result)  # Should be expanded

    @patch("os.path.expanduser")
    def test_cache_root_uses_expanduser(self, mock_expanduser):
        """Test that _cache_root uses os.path.expanduser."""
        mock_expanduser.return_value = "/home/user/.openbench"
        result = _cache_root()
        mock_expanduser.assert_called_once_with("~/.openbench")
        assert str(result).endswith(".openbench")


class TestCacheDiscovery:
    """Test cache discovery functions."""

    @patch("openbench._cli.cache_command._cache_root")
    def test_discover_cache_types_no_root(self, mock_cache_root):
        """Test _discover_cache_types when root doesn't exist."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_cache_root.return_value = mock_path

        result = _discover_cache_types()
        assert result == []

    @patch("openbench._cli.cache_command._cache_root")
    def test_discover_cache_types_with_dirs(self, mock_cache_root):
        """Test _discover_cache_types with multiple cache directories."""
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


class TestHumanSize:
    """Test _human_size function."""

    def test_human_size_bytes(self):
        """Test formatting bytes."""
        assert _human_size(0) == "0.0 B"
        assert _human_size(1) == "1.0 B"
        assert _human_size(512) == "512.0 B"
        assert _human_size(1023) == "1023.0 B"

    def test_human_size_kilobytes(self):
        """Test formatting kilobytes."""
        assert _human_size(1024) == "1.0 KB"
        assert _human_size(1536) == "1.5 KB"
        assert _human_size(2048) == "2.0 KB"
        assert _human_size(1024 * 1023) == "1023.0 KB"

    def test_human_size_megabytes(self):
        """Test formatting megabytes."""
        assert _human_size(1024 * 1024) == "1.0 MB"
        assert _human_size(1024 * 1024 * 1.5) == "1.5 MB"
        assert _human_size(1024 * 1024 * 1023) == "1023.0 MB"

    def test_human_size_gigabytes(self):
        """Test formatting gigabytes."""
        assert _human_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _human_size(1024 * 1024 * 1024 * 2.5) == "2.5 GB"

    def test_human_size_terabytes(self):
        """Test formatting terabytes."""
        assert _human_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"
        assert _human_size(1024 * 1024 * 1024 * 1024 * 3.7) == "3.7 TB"

    def test_human_size_petabytes(self):
        """Test formatting petabytes."""
        size = 1024 * 1024 * 1024 * 1024 * 1024
        assert _human_size(size) == "1.0 PB"
        assert _human_size(size * 2.2) == "2.2 PB"

    def test_human_size_edge_cases(self):
        """Test edge cases for human size formatting."""
        # Very small float
        assert _human_size(0.5) == "0.5 B"

        # Exact boundaries
        assert _human_size(1023.9) == "1023.9 B"
        assert _human_size(1024.0) == "1.0 KB"


class TestDirSize:
    """Test _dir_size function."""

    def test_dir_size_nonexistent_path(self):
        """Test _dir_size with nonexistent path."""
        nonexistent = Path("/this/path/does/not/exist")
        assert _dir_size(nonexistent) == 0

    def test_dir_size_empty_directory(self):
        """Test _dir_size with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert _dir_size(Path(temp_dir)) == 0

    def test_dir_size_single_file_as_path(self):
        """Test _dir_size when path points to a single file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = b"hello world"
            temp_file.write(content)
            temp_file.flush()

            try:
                result = _dir_size(Path(temp_file.name))
                assert result == len(content)
            finally:
                os.unlink(temp_file.name)

    def test_dir_size_directory_with_files(self):
        """Test _dir_size with directory containing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with known content
            file1 = temp_path / "file1.txt"
            file1.write_bytes(b"a" * 100)

            file2 = temp_path / "file2.txt"
            file2.write_bytes(b"b" * 200)

            assert _dir_size(temp_path) == 300

    def test_dir_size_nested_directories(self):
        """Test _dir_size with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            (temp_path / "file1.txt").write_bytes(b"x" * 50)

            subdir1 = temp_path / "subdir1"
            subdir1.mkdir()
            (subdir1 / "file2.txt").write_bytes(b"y" * 75)

            subdir2 = subdir1 / "nested"
            subdir2.mkdir()
            (subdir2 / "file3.txt").write_bytes(b"z" * 25)

            assert _dir_size(temp_path) == 150

    def test_dir_size_ignores_stat_errors(self):
        """Test that _dir_size ignores files that can't be stat'd."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "good_file.txt").write_bytes(b"test")

            # Mock rglob to return both a good file and one that raises an exception
            good_file = temp_path / "good_file.txt"
            bad_file = MagicMock()
            bad_file.is_file.return_value = True
            bad_file.stat.side_effect = PermissionError("Access denied")

            with patch.object(Path, "rglob", return_value=[good_file, bad_file]):
                # Should not raise exception, should return size of good file only
                result = _dir_size(temp_path)
                assert result == 4  # Length of b"test"

    def test_dir_size_with_symlinks_and_special_files(self):
        """Test _dir_size handles various file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Regular file
            regular_file = temp_path / "regular.txt"
            regular_file.write_bytes(b"regular" * 10)

            # Create a subdirectory that should be skipped by is_file() check
            subdir = temp_path / "subdir"
            subdir.mkdir()

            result = _dir_size(temp_path)
            assert result == 70  # 7 * 10


class TestPrintTree:
    """Test _print_tree function."""

    @patch("typer.echo")
    def test_print_tree_empty_directory(self, mock_echo):
        """Test _print_tree with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            _print_tree(Path(temp_dir))
            # Should not print anything for empty directory
            mock_echo.assert_not_called()

    @patch("typer.echo")
    def test_print_tree_single_file(self, mock_echo):
        """Test _print_tree with single file in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")

            _print_tree(temp_path)

            mock_echo.assert_called_once_with("└── test.txt")

    @patch("typer.echo")
    def test_print_tree_multiple_files(self, mock_echo):
        """Test _print_tree with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "a.txt").write_text("content")
            (temp_path / "b.txt").write_text("content")

            _print_tree(temp_path)

            # Should be called twice, once for each file
            assert mock_echo.call_count == 2
            # First file should use ├── and second should use └──
            calls = [call.args[0] for call in mock_echo.call_args_list]
            assert "├── a.txt" in calls
            assert "└── b.txt" in calls

    @patch("typer.echo")
    def test_print_tree_directories_first(self, mock_echo):
        """Test _print_tree shows directories before files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "z_file.txt").write_text("content")
            (temp_path / "a_directory").mkdir()

            _print_tree(temp_path)

            calls = [call.args[0] for call in mock_echo.call_args_list]
            # Directory should come first despite alphabetical order
            assert calls[0] == "├── a_directory"
            assert calls[1] == "└── z_file.txt"

    @patch("typer.echo")
    def test_print_tree_recursive(self, mock_echo):
        """Test _print_tree recursively prints subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("content")

            _print_tree(temp_path)

            calls = [call.args[0] for call in mock_echo.call_args_list]
            assert any("└── subdir" in call for call in calls)
            assert any("    └── nested.txt" in call for call in calls)

    @patch("typer.echo")
    def test_print_tree_complex_structure(self, mock_echo):
        """Test _print_tree with complex directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create complex structure
            (temp_path / "file1.txt").write_text("content")

            dir1 = temp_path / "dir1"
            dir1.mkdir()
            (dir1 / "file2.txt").write_text("content")

            dir2 = temp_path / "dir2"
            dir2.mkdir()
            nested = dir2 / "nested"
            nested.mkdir()
            (nested / "deep.txt").write_text("content")

            _print_tree(temp_path)

            # Verify structure is printed correctly
            calls = [call.args[0] for call in mock_echo.call_args_list]

            # Should have proper tree structure with correct connectors
            assert any("├── dir1" in call for call in calls)
            assert any("├── dir2" in call for call in calls)
            assert any("└── file1.txt" in call for call in calls)
            assert any("│   └── nested" in call for call in calls)
            assert any("    └── deep.txt" in call for call in calls)

    @patch("typer.echo")
    def test_print_tree_file_not_found(self, mock_echo):
        """Test _print_tree handles FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path")
        _print_tree(nonexistent_path)

        mock_echo.assert_called_once_with(f"Path not found: {nonexistent_path}")

    @patch("typer.echo")
    def test_print_tree_with_prefix(self, mock_echo):
        """Test _print_tree respects prefix parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.txt").write_text("content")

            _print_tree(temp_path, prefix="  ")

            mock_echo.assert_called_once_with("  └── test.txt")


class TestCacheCommandEdgeCases:
    """Test edge cases and error conditions."""

    def test_human_size_negative_number(self):
        """Test _human_size with negative number."""
        # Should handle negative numbers (though not expected in real usage)
        # The function doesn't convert negative numbers, it just formats them as bytes
        result = _human_size(-1024)
        assert result == "-1024.0 B"

    def test_human_size_zero(self):
        """Test _human_size with zero."""
        assert _human_size(0) == "0.0 B"

    def test_human_size_very_large_number(self):
        """Test _human_size with extremely large number."""
        very_large = 1024**6  # Exabyte
        result = _human_size(very_large)
        assert result == "1024.0 PB"  # Should still use PB as largest unit

    @patch("pathlib.Path.exists")
    def test_dir_size_path_disappears_during_calculation(self, mock_exists):
        """Test _dir_size when path disappears during calculation."""
        mock_exists.return_value = False
        result = _dir_size(Path("/some/path"))
        assert result == 0

    def test_cache_root_consistency(self):
        """Test that _cache_root returns consistent results."""
        result1 = _cache_root()
        result2 = _cache_root()
        assert result1 == result2
        assert str(result1) == str(result2)
