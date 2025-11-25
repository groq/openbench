"""
Shared path utilities for ProgressiveMCPBench.

These helpers keep sandbox path rewriting consistent across strategies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def root_sandbox_dir() -> Path:
    """Return the root sandbox directory used by ProgressiveMCPBench tools."""
    return Path(os.path.expanduser("~/.openbench/progressivemcpbench/root")).resolve()


def rewrite_root_path(value: str) -> str:
    """Rewrite /root/... paths to the sandbox location."""
    if value.startswith("/root/"):
        base = root_sandbox_dir()
        rel = value[len("/root/") :]
        return str(base / rel)
    return value


def rewrite_params_for_root(obj: Any) -> Any:
    """Recursively rewrite any /root/... strings inside tool parameters."""
    if isinstance(obj, dict):
        return {k: rewrite_params_for_root(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_params_for_root(v) for v in obj]
    if isinstance(obj, str):
        new_path = rewrite_root_path(obj)
        if new_path != obj:
            # Ensure parent dir exists for prospective outputs
            p = Path(new_path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return new_path
    return obj
