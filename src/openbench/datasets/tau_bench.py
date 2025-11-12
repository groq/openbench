"""
Dataset helpers for tau-bench domains.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from inspect_ai.dataset import MemoryDataset, Sample

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

TAU2_REPO_URL = "https://github.com/sierra-research/tau2-bench.git"
TAU2_DATA_ENV = "TAU2_DATA_DIR"
DEFAULT_TAU2_DATA_DIR = Path("~/.openbench/tau2").expanduser()


def _import_tau2_get_tasks():
    """
    Import tau2.run.get_tasks lazily so TAU2_DATA_DIR can be configured first.
    """
    from tau2.run import get_tasks as tau2_get_tasks  # type: ignore

    return tau2_get_tasks


def _serialize_task(task) -> dict:
    """
    Convert a tau2 Task (pydantic model) into a JSON-serializable dict.
    """
    return task.model_dump(mode="json")


def _task_prompt(task) -> str:
    """
    Extract a human-readable prompt from the task's user scenario.
    """
    scenario = getattr(task, "user_scenario", None)
    if scenario is None:
        return f"TauBench task {task.id}"
    instructions = getattr(scenario, "instructions", None)
    if instructions is None:
        return f"TauBench task {task.id}"
    return str(instructions)


def _download_tau2_data(target: Path) -> None:
    """Clone tau2-bench and copy its data directory into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="openbench_tau2_"))
    repo_dir = tmp_dir / "tau2-bench"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                "main",
                TAU2_REPO_URL,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
        )
        data_dir = repo_dir / "data"
        if not data_dir.exists():
            raise ValueError(
                f"Downloaded repository is missing the data directory at {data_dir}"
            )
        shutil.copytree(data_dir, target, dirs_exist_ok=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external git
        raise ValueError(
            f"Failed to download tau2 assets from {TAU2_REPO_URL}: {exc.stderr.decode().strip()}"
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_tau2_data_dir() -> Path:
    """
    Ensure TAU2_DATA_DIR points at a directory containing tau2's data assets.
    Downloads from the official repository if necessary.
    """
    data_dir_env = os.getenv(TAU2_DATA_ENV)
    if data_dir_env:
        path = Path(data_dir_env).expanduser()
        if not path.exists():
            raise ValueError(
                f"{TAU2_DATA_ENV}={path} does not exist. "
                "Either point it at a valid tau2 data checkout or unset it so "
                "openbench can download the assets."
            )
        return path

    target = DEFAULT_TAU2_DATA_DIR
    sentinel = target / "tau2" / "domains"
    if not sentinel.exists():
        _download_tau2_data(target)
    os.environ[TAU2_DATA_ENV] = str(target)
    return target


def get_tau_bench_dataset(
    domain: str,
    *,
    num_trials: int = 1,
    task_ids: Optional[Iterable[str]] = None,
    num_tasks: Optional[int] = None,
) -> MemoryDataset:
    """
    Load tau2 tasks for a domain and expose them as an Inspect dataset.

    Args:
        domain: tau2 domain name (retail, airline, telecom, etc.).
        num_trials: Number of times to repeat each task.
        task_ids: Optional subset of task ids.
        num_tasks: Optional slice of the task list (after filtering).
    """
    ensure_tau2_data_dir()

    get_tasks = _import_tau2_get_tasks()

    tasks: List = get_tasks(
        domain, task_ids=list(task_ids) if task_ids else None, num_tasks=num_tasks
    )  # type: ignore[arg-type]
    samples: list[Sample] = []
    for task in tasks:
        serialized = _serialize_task(task)
        prompt = _task_prompt(task)
        for trial in range(1, num_trials + 1):
            samples.append(
                Sample(
                    id=f"{domain}-{task.id}-trial{trial}",
                    input=prompt,
                    target="tau_bench",
                    metadata={
                        "domain": domain,
                        "tau2_task": serialized,
                        "trial": trial,
                    },
                )
            )
    return MemoryDataset(samples=samples, name=f"tau_bench_{domain}")
