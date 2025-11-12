import subprocess
import sys

from inspect_ai.scorer._reducer.registry import create_reducers


def test_pass_hat_reducer_available_in_current_process():
    import openbench  # noqa: F401

    reducers = create_reducers("pass_hat_1")
    assert reducers is not None
    assert len(reducers) == 1


def test_pass_hat_reducer_registered_via_cli_entrypoint(tmp_path):
    script = (
        "import openbench._cli\n"
        "from inspect_ai.scorer._reducer.registry import create_reducers\n"
        "reducers = create_reducers('pass_hat_2')\n"
        "assert reducers and len(reducers) == 1\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(
            "pass_hat reducer not registered via CLI import:\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
