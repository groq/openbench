"""Unit tests for cache utility functions."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from openbench.utils.cache import (
    prepare_livemcpbench_cache,
    _livemcpbench_root_dir,
    clear_livemcpbench_root,
)


class TestPrepareLivemcpbenchCache:
    """Test the prepare_livemcpbench_cache function."""

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch("openbench.utils.cache.typer.secho")
    @patch("openbench.utils.cache.typer.echo")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_successful_preparation(
        self, mock_echo, mock_secho, mock_prepare_root, mock_prepare_copilot
    ):
        """Test successful cache preparation with required environment variable."""
        # Setup mocks
        mock_cache_path = Path("/test/cache/path")
        mock_root_path = Path("/test/root/path")
        mock_prepare_copilot.return_value = mock_cache_path
        mock_prepare_root.return_value = mock_root_path

        # Call function
        result = prepare_livemcpbench_cache()

        # Verify results
        assert result == mock_root_path
        assert os.environ["MCP_DATA_PATH"] == str(mock_cache_path)

        # Verify function calls
        mock_secho.assert_called_once_with(
            "\n🔧 Preparing LiveMCPBench caches...", fg=typer.colors.CYAN
        )
        mock_prepare_copilot.assert_called_once_with(
            force_refresh=False, embeddings_path=None
        )
        mock_prepare_root.assert_called_once_with(force_refresh=False)
        mock_echo.assert_any_call(f"  ✅ Embedding cache ready: {mock_cache_path}")
        mock_echo.assert_any_call(f"  ✅ Root sandbox ready: {mock_root_path}\n")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_openai_api_key(self):
        """Test that RuntimeError is raised when OPENAI_API_KEY is missing."""
        with pytest.raises(RuntimeError) as exc_info:
            prepare_livemcpbench_cache()

        assert "OPENAI_API_KEY is required for LiveMCPBench" in str(exc_info.value)

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch("openbench.utils.cache.typer.secho")
    @patch("openbench.utils.cache.typer.echo")
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_empty_openai_api_key(
        self, mock_echo, mock_secho, mock_prepare_root, mock_prepare_copilot
    ):
        """Test that empty OPENAI_API_KEY is treated as missing."""
        with pytest.raises(RuntimeError) as exc_info:
            prepare_livemcpbench_cache()

        assert "OPENAI_API_KEY is required for LiveMCPBench" in str(exc_info.value)

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch("openbench.utils.cache.typer.secho")
    @patch("openbench.utils.cache.typer.echo")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_mcp_data_path_environment_variable(
        self, mock_echo, mock_secho, mock_prepare_root, mock_prepare_copilot
    ):
        """Test that MCP_DATA_PATH is correctly set in environment."""
        # Setup mocks
        mock_cache_path = Path("/custom/cache/path/embeddings.json")
        mock_root_path = Path("/custom/root/path")
        mock_prepare_copilot.return_value = mock_cache_path
        mock_prepare_root.return_value = mock_root_path

        # Ensure MCP_DATA_PATH is not set initially
        if "MCP_DATA_PATH" in os.environ:
            del os.environ["MCP_DATA_PATH"]

        # Call function
        prepare_livemcpbench_cache()

        # Verify MCP_DATA_PATH was set correctly
        assert os.environ["MCP_DATA_PATH"] == str(mock_cache_path)

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_prepare_copilot_cache_exception(
        self, mock_prepare_root, mock_prepare_copilot
    ):
        """Test handling of exceptions from prepare_copilot_cache."""
        mock_prepare_copilot.side_effect = RuntimeError("Copilot cache error")

        with pytest.raises(RuntimeError, match="Copilot cache error"):
            prepare_livemcpbench_cache()

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_prepare_root_data_exception(self, mock_prepare_root, mock_prepare_copilot):
        """Test handling of exceptions from prepare_root_data."""
        mock_cache_path = Path("/test/cache/path")
        mock_prepare_copilot.return_value = mock_cache_path
        mock_prepare_root.side_effect = RuntimeError("Root data error")

        with pytest.raises(RuntimeError, match="Root data error"):
            prepare_livemcpbench_cache()


class TestLivemcpbenchRootDir:
    """Test the _livemcpbench_root_dir function."""

    @patch("openbench.utils.cache.os.path.expanduser")
    def test_root_dir_path_expansion(self, mock_expanduser):
        """Test that the function correctly expands the user path."""
        mock_expanduser.return_value = "/home/testuser/.openbench/livemcpbench/root"

        result = _livemcpbench_root_dir()

        mock_expanduser.assert_called_once_with("~/.openbench/livemcpbench/root")
        assert result == Path("/home/testuser/.openbench/livemcpbench/root").resolve()

    @patch("openbench.utils.cache.os.path.expanduser")
    def test_root_dir_path_resolution(self, mock_expanduser):
        """Test that the function resolves the path correctly."""
        # Use a real-looking path to test resolution
        mock_expanduser.return_value = "/Users/testuser/.openbench/livemcpbench/root"

        result = _livemcpbench_root_dir()

        assert isinstance(result, Path)
        assert result.is_absolute()
        # Verify it's resolved (no relative components)
        assert str(result) == str(result.resolve())

    def test_root_dir_default_path(self):
        """Test the default path structure."""
        result = _livemcpbench_root_dir()

        # Should always end with the same path structure
        assert str(result).endswith("/.openbench/livemcpbench/root")
        assert isinstance(result, Path)
        assert result.is_absolute()


class TestClearLivemcpbenchRoot:
    """Test the clear_livemcpbench_root function."""

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_successful_cleanup_with_existing_directory(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test successful cleanup when directory exists."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_path.__str__ = Mock(return_value="/test/root/path")
        mock_root_dir.return_value = mock_root_path

        clear_livemcpbench_root(quiet=False)

        mock_rmtree.assert_called_once_with(mock_root_path)
        mock_echo.assert_called_once_with(
            "🧹 Cleaned LiveMCPBench root: /test/root/path"
        )

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_with_nonexistent_directory(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test cleanup when directory doesn't exist."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = False
        mock_root_path.__str__ = Mock(return_value="/test/root/path")
        mock_root_dir.return_value = mock_root_path

        clear_livemcpbench_root(quiet=False)

        mock_rmtree.assert_not_called()
        mock_echo.assert_called_once_with(
            "(LiveMCPBench root already clean: /test/root/path)"
        )

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_quiet_mode_existing_directory(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test quiet mode with existing directory."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_dir.return_value = mock_root_path

        clear_livemcpbench_root(quiet=True)

        mock_rmtree.assert_called_once_with(mock_root_path)
        mock_echo.assert_not_called()

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_quiet_mode_nonexistent_directory(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test quiet mode with nonexistent directory."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = False
        mock_root_dir.return_value = mock_root_path

        clear_livemcpbench_root(quiet=True)

        mock_rmtree.assert_not_called()
        mock_echo.assert_not_called()

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_exception_handling(self, mock_echo, mock_rmtree, mock_root_dir):
        """Test exception handling during cleanup."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_path.__str__ = Mock(return_value="/test/root/path")
        mock_root_dir.return_value = mock_root_path
        mock_rmtree.side_effect = PermissionError("Permission denied")

        # Should not raise exception
        clear_livemcpbench_root(quiet=False)

        mock_rmtree.assert_called_once_with(mock_root_path)
        mock_echo.assert_called_once_with(
            "⚠️  Failed to clean LiveMCPBench root (/test/root/path): Permission denied"
        )

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_exception_handling_quiet_mode(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test exception handling during cleanup in quiet mode."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_dir.return_value = mock_root_path
        mock_rmtree.side_effect = PermissionError("Permission denied")

        # Should not raise exception or print anything
        clear_livemcpbench_root(quiet=True)

        mock_rmtree.assert_called_once_with(mock_root_path)
        mock_echo.assert_not_called()

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_default_quiet_parameter(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test default value of quiet parameter."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_path.__str__ = Mock(return_value="/test/root/path")
        mock_root_dir.return_value = mock_root_path

        # Call without quiet parameter (should default to False)
        clear_livemcpbench_root()

        mock_rmtree.assert_called_once_with(mock_root_path)
        mock_echo.assert_called_once_with(
            "🧹 Cleaned LiveMCPBench root: /test/root/path"
        )

    @patch("openbench.utils.cache._livemcpbench_root_dir")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.echo")
    def test_cleanup_multiple_exception_types(
        self, mock_echo, mock_rmtree, mock_root_dir
    ):
        """Test handling of different exception types."""
        mock_root_path = Mock(spec=Path)
        mock_root_path.exists.return_value = True
        mock_root_path.__str__ = Mock(return_value="/test/root/path")
        mock_root_dir.return_value = mock_root_path

        # Test with different exception types
        exceptions = [
            FileNotFoundError("File not found"),
            OSError("OS error"),
            Exception("Generic error"),
        ]

        for exc in exceptions:
            mock_rmtree.side_effect = exc
            mock_echo.reset_mock()

            clear_livemcpbench_root(quiet=False)

            mock_echo.assert_called_once_with(
                f"⚠️  Failed to clean LiveMCPBench root (/test/root/path): {exc}"
            )


class TestCacheIntegration:
    """Integration tests for cache functions working together."""

    @patch("openbench.utils.cache.prepare_copilot_cache")
    @patch("openbench.utils.cache.prepare_root_data")
    @patch("openbench.utils.cache.shutil.rmtree")
    @patch("openbench.utils.cache.typer.secho")
    @patch("openbench.utils.cache.typer.echo")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_prepare_then_clear_workflow(
        self,
        mock_echo,
        mock_secho,
        mock_rmtree,
        mock_prepare_root,
        mock_prepare_copilot,
    ):
        """Test the typical workflow of preparing then clearing cache."""
        # Setup mocks for preparation
        mock_cache_path = Path("/test/cache/path")
        mock_root_path = Path("/test/root/path")
        mock_prepare_copilot.return_value = mock_cache_path
        mock_prepare_root.return_value = mock_root_path

        # Prepare cache
        result = prepare_livemcpbench_cache()
        assert result == mock_root_path
        assert os.environ["MCP_DATA_PATH"] == str(mock_cache_path)

        # Mock the root directory for cleanup
        with patch("openbench.utils.cache._livemcpbench_root_dir") as mock_root_dir:
            mock_root_dir_obj = Mock(spec=Path)
            mock_root_dir_obj.exists.return_value = True
            mock_root_dir_obj.__str__ = Mock(return_value=str(mock_root_path))
            mock_root_dir.return_value = mock_root_dir_obj

            # Clear cache
            clear_livemcpbench_root(quiet=False)

            # Verify cleanup was called
            mock_rmtree.assert_called_once_with(mock_root_dir_obj)

    @patch("openbench.utils.cache.os.path.expanduser")
    def test_root_dir_consistency(self, mock_expanduser):
        """Test that _livemcpbench_root_dir returns consistent paths."""
        mock_expanduser.return_value = "/home/user/.openbench/livemcpbench/root"

        # Call multiple times and verify consistency
        path1 = _livemcpbench_root_dir()
        path2 = _livemcpbench_root_dir()

        assert path1 == path2
        assert str(path1) == str(path2)
        # Verify expanduser was called each time
        assert mock_expanduser.call_count == 2
