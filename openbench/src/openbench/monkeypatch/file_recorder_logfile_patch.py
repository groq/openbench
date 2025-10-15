"""
Monkey patch for inspect_ai FileRecorder to allow setting a custom logfile name.

Usage:
    from openbench.monkeypatch.file_recorder_logfile_patch import patch_file_recorder_logfile
    patch_file_recorder_logfile(logfile)

Call this before invoking inspect_ai.eval().
"""


def patch_file_recorder_logfile(logfile: str):
    """
    Monkey patch FileRecorder._log_file_key to always use the provided logfile name.
    Args:
        logfile: The desired logfile name (without extension).
    """
    import inspect_ai.log._recorders.file as file_recorder_mod

    def custom_log_file_key(self, eval_spec):
        return logfile

    file_recorder_mod.FileRecorder._log_file_key = custom_log_file_key  # type: ignore[method-assign]
