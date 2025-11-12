from ._cli.eval_command import run_eval

# Ensure custom Inspect reducers (e.g., pass_hat) are registered at import time.
from .scorers.pass_hat import pass_hat as _register_pass_hat  # noqa: F401

__all__ = ["run_eval"]
