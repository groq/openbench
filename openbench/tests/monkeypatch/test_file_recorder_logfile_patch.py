"""Test the file recorder logfile monkey patch."""

import pytest
from unittest.mock import patch, MagicMock
from openbench.monkeypatch.file_recorder_logfile_patch import (
    patch_file_recorder_logfile,
)


@pytest.fixture
def mock_file_recorder_module():
    """Mock the inspect_ai.log._recorders.file module."""
    with patch("inspect_ai.log._recorders.file") as mock_module:
        # Set up the FileRecorder class within the module
        mock_module.FileRecorder = MagicMock()
        yield mock_module


def test_patch_file_recorder_logfile(mock_file_recorder_module):
    """Test that patch_file_recorder_logfile correctly patches the FileRecorder."""
    test_logfile = "test_logfile"

    # Apply the patch
    patch_file_recorder_logfile(test_logfile)

    # Verify that _log_file_key was assigned a new function
    assert mock_file_recorder_module.FileRecorder._log_file_key is not None

    # Create a mock instance and test the patched method
    mock_instance = MagicMock()
    mock_eval_spec = MagicMock()

    # Get the patched method and call it
    patched_method = mock_file_recorder_module.FileRecorder._log_file_key
    result = patched_method(mock_instance, mock_eval_spec)

    # Verify it returns the expected logfile name
    assert result == test_logfile
