"""Unit tests for FActScore database download functionality."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openbench.utils.factscore_cache import (
    download_factscore_db,
    _check_disk_space,
    _compute_file_hash,
)


class TestCheckDiskSpace:
    """Test disk space checking utility."""

    @patch("shutil.disk_usage")
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check returns True when sufficient space available."""
        mock_stat = MagicMock()
        mock_stat.free = 30 * 1024 * 1024 * 1024  # 30GB free
        mock_disk_usage.return_value = mock_stat

        required = 22 * 1024 * 1024 * 1024  # 22GB required
        result = _check_disk_space(Path("/tmp"), required)

        assert result is True
        mock_disk_usage.assert_called_once_with(Path("/tmp"))

    @patch("shutil.disk_usage")
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check returns False when insufficient space available."""
        mock_stat = MagicMock()
        mock_stat.free = 10 * 1024 * 1024 * 1024  # 10GB free
        mock_disk_usage.return_value = mock_stat

        required = 22 * 1024 * 1024 * 1024  # 22GB required
        result = _check_disk_space(Path("/tmp"), required)

        assert result is False

    @patch("shutil.disk_usage")
    def test_check_disk_space_exact_amount(self, mock_disk_usage):
        """Test disk space check with exactly required amount."""
        mock_stat = MagicMock()
        mock_stat.free = 22 * 1024 * 1024 * 1024  # Exactly 22GB
        mock_disk_usage.return_value = mock_stat

        required = 22 * 1024 * 1024 * 1024  # 22GB required
        result = _check_disk_space(Path("/tmp"), required)

        assert result is True


class TestComputeFileHash:
    """Test file hash computation utility."""

    def test_compute_file_hash_sha256(self):
        """Test SHA-256 hash computation for a file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = b"test content for hashing"
            temp_file.write(content)
            temp_file.flush()

            try:
                result = _compute_file_hash(Path(temp_file.name))

                # Compute expected hash
                expected = hashlib.sha256(content).hexdigest()

                assert result == expected
                assert len(result) == 64  # SHA-256 produces 64-character hex
            finally:
                Path(temp_file.name).unlink()

    def test_compute_file_hash_empty_file(self):
        """Test hash computation for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.flush()

            try:
                result = _compute_file_hash(Path(temp_file.name))

                # Hash of empty file
                expected = hashlib.sha256(b"").hexdigest()

                assert result == expected
            finally:
                Path(temp_file.name).unlink()

    def test_compute_file_hash_large_file(self):
        """Test hash computation reads file in chunks efficiently."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Create a file larger than one chunk (8192 bytes)
            content = b"x" * 20000
            temp_file.write(content)
            temp_file.flush()

            try:
                result = _compute_file_hash(Path(temp_file.name))

                # Compute expected hash
                expected = hashlib.sha256(content).hexdigest()

                assert result == expected
            finally:
                Path(temp_file.name).unlink()


class TestDownloadFactScoreDB:
    """Test FActScore database download function."""

    @patch("openbench.utils.factscore_cache.Console")
    def test_download_skips_if_file_exists(self, mock_console_class):
        """Test download is skipped if database file already exists."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create existing database file
            db_file = Path(temp_dir) / "data" / "enwiki-20230401.db"
            db_file.parent.mkdir(parents=True)
            db_file.write_bytes(b"existing database" * 1000)

            with patch(
                "openbench.utils.factscore_cache.data_dir", return_value=db_file.parent
            ):
                result = download_factscore_db()

                assert result == db_file
                # Should print message about existing file
                assert mock_console.print.called
                print_args = str(mock_console.print.call_args_list)
                assert "already exists" in print_args

    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_raises_error_insufficient_disk_space(
        self, mock_console_class, mock_check_disk
    ):
        """Test download raises OSError when insufficient disk space."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with pytest.raises(OSError) as exc_info:
                    download_factscore_db()

                assert "Insufficient disk space" in str(exc_info.value)

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_successful_without_hash_verification(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test successful download without hash verification."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            # Mock hf_hub_download to create a fake downloaded file
            def fake_hf_download(repo_id, filename, repo_type, local_dir):
                # hf_hub_download creates the file at local_dir/filename
                downloaded_file = Path(local_dir) / filename
                downloaded_file.write_bytes(b"fake database content" * 1000)
                return str(downloaded_file)

            mock_hf_download.side_effect = fake_hf_download

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                result = download_factscore_db(expected_sha256=None)

                assert result == db_dir / "enwiki-20230401.db"
                assert result.exists()
                assert result.read_bytes() == b"fake database content" * 1000

                # Verify hf_hub_download was called with correct parameters
                mock_hf_download.assert_called_once()
                call_kwargs = mock_hf_download.call_args.kwargs
                assert call_kwargs["repo_id"] == "lvogel123/factscore-data"
                assert call_kwargs["filename"] == "enwiki-20230401.db"
                assert call_kwargs["repo_type"] == "dataset"

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_successful_with_hash_verification(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test successful download with hash verification."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        fake_content = b"fake database content" * 1000
        expected_hash = hashlib.sha256(fake_content).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            # Mock hf_hub_download to create a fake downloaded file
            def fake_hf_download(repo_id, filename, repo_type, local_dir):
                downloaded_file = Path(local_dir) / filename
                downloaded_file.write_bytes(fake_content)
                return str(downloaded_file)

            mock_hf_download.side_effect = fake_hf_download

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                result = download_factscore_db(expected_sha256=expected_hash)

                assert result == db_dir / "enwiki-20230401.db"
                assert result.exists()
                # Verify hash verification message was printed
                print_calls = [str(call) for call in mock_console.print.call_args_list]
                assert any(
                    "integrity verified" in str(call).lower() for call in print_calls
                )

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_fails_hash_verification(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download fails when hash verification doesn't match."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        fake_content = b"fake database content"
        wrong_hash = "a" * 64  # Wrong hash

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            # Mock hf_hub_download to create a fake downloaded file
            def fake_hf_download(repo_id, filename, repo_type, local_dir):
                downloaded_file = Path(local_dir) / filename
                downloaded_file.write_bytes(fake_content)
                return str(downloaded_file)

            mock_hf_download.side_effect = fake_hf_download

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with pytest.raises(RuntimeError) as exc_info:
                    download_factscore_db(expected_sha256=wrong_hash, max_retries=1)

                assert "integrity check failed" in str(exc_info.value).lower()

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_retries_on_failure(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download retries on failure."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        # First two attempts fail, third succeeds
        call_count = [0]

        def fake_hf_download(repo_id, filename, repo_type, local_dir):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            downloaded_file = Path(local_dir) / filename
            downloaded_file.write_bytes(b"fake database")
            return str(downloaded_file)

        mock_hf_download.side_effect = fake_hf_download

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with patch("time.sleep"):  # Skip sleep delays in tests
                    result = download_factscore_db(max_retries=3, expected_sha256=None)

                    assert result.exists()
                    assert mock_hf_download.call_count == 3
                    # Verify retry messages were printed
                    print_calls = [
                        str(call) for call in mock_console.print.call_args_list
                    ]
                    assert any("retry" in str(call).lower() for call in print_calls)

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_fails_after_max_retries(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download raises error after max retries exhausted."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        # All attempts fail
        mock_hf_download.side_effect = ConnectionError("Network error")

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with patch("time.sleep"):  # Skip sleep delays in tests
                    with pytest.raises(RuntimeError) as exc_info:
                        download_factscore_db(max_retries=2)

                    assert "Failed to download" in str(exc_info.value)
                    assert "2 attempts" in str(exc_info.value)
                    assert mock_hf_download.call_count == 2

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_cleans_up_temp_files_on_failure(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download cleans up temporary files on failure."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        # Fail download
        mock_hf_download.side_effect = RuntimeError("Download failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with patch("time.sleep"):  # Skip sleep delays
                    with pytest.raises(RuntimeError):
                        download_factscore_db(max_retries=1)

                    # Check that no factscore_ temp directories remain
                    temp_dirs = [
                        d for d in db_dir.iterdir() if d.name.startswith("factscore_")
                    ]
                    assert len(temp_dirs) == 0

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_handles_empty_file(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download detects and fails on empty downloaded file."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        # Create empty file
        def fake_hf_download(repo_id, filename, repo_type, local_dir):
            downloaded_file = Path(local_dir) / filename
            downloaded_file.write_bytes(b"")
            return str(downloaded_file)

        mock_hf_download.side_effect = fake_hf_download

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                with patch("time.sleep"):
                    with pytest.raises(RuntimeError) as exc_info:
                        download_factscore_db(max_retries=1)

                    assert "empty" in str(exc_info.value).lower()

    @patch("openbench.utils.factscore_cache.hf_hub_download")
    @patch("openbench.utils.factscore_cache._check_disk_space")
    @patch("openbench.utils.factscore_cache.Console")
    def test_download_atomic_write(
        self, mock_console_class, mock_check_disk, mock_hf_download
    ):
        """Test download uses atomic write (temp dir then move)."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        mock_check_disk.return_value = True

        downloaded_to_temp = [False]

        def fake_hf_download(repo_id, filename, repo_type, local_dir):
            # Verify download happens to temp location, not final location
            local_dir_path = Path(local_dir)
            if "factscore_" in local_dir_path.name:
                downloaded_to_temp[0] = True
            downloaded_file = local_dir_path / filename
            downloaded_file.write_bytes(b"database content")
            return str(downloaded_file)

        mock_hf_download.side_effect = fake_hf_download

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "data"
            db_dir.mkdir()

            with patch("openbench.utils.factscore_cache.data_dir", return_value=db_dir):
                result = download_factscore_db(expected_sha256=None)

                # Verify download happened to temp location first
                assert downloaded_to_temp[0]

                # Verify final file exists at correct location
                assert result == db_dir / "enwiki-20230401.db"
                assert result.exists()

    @patch("openbench.utils.factscore_cache.Console")
    def test_download_displays_file_size(self, mock_console_class):
        """Test download displays file size information."""
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create existing database file with known size
            db_file = Path(temp_dir) / "data" / "enwiki-20230401.db"
            db_file.parent.mkdir(parents=True)
            # Create 2GB file
            size_2gb = 2 * 1024 * 1024 * 1024
            db_file.write_bytes(b"x" * min(size_2gb, 10000))  # Actual size for test

            with patch(
                "openbench.utils.factscore_cache.data_dir", return_value=db_file.parent
            ):
                # Mock the stat to return 2GB
                with patch.object(Path, "stat") as mock_stat:
                    mock_stat_obj = MagicMock()
                    mock_stat_obj.st_size = size_2gb
                    mock_stat.return_value = mock_stat_obj

                    download_factscore_db()

                    # Verify file size was printed
                    print_calls = [
                        str(call) for call in mock_console.print.call_args_list
                    ]
                    assert any("GB" in str(call) for call in print_calls)
