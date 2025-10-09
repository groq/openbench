"""
Adapted from CVE-bench: https://github.com/uiuc-kang-lab/cve-bench
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from inspect_ai import Task, task

from openbench.datasets.cvebench import (
    create_dataset,
    filter_dataset_by_variant,
    resolve_challenges_dir,
)
from openbench.scorers.cvebench import STORE_KEY_ATTACK_SUCCESSFUL, check_store
from openbench.solvers.cvebench import default_solver


def _make_absolute(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _resolve_challenge_paths(
    challenges: str | Iterable[str] | None,
    base_dir: Path,
) -> list[Path]:
    if challenges is None:
        return [base_dir]

    if isinstance(challenges, str):
        challenges = [challenges]

    return [_make_absolute(ch, base_dir) for ch in challenges]


@task
def cvebench(
    challenges: str | list[str] | None = None,
    variants: str | list[str] | None = None,
    max_messages: int = 50,
    challenges_dir: str | Path | None = None,
) -> Task:
    """Construct the CVEBench Task."""

    base_dir = resolve_challenges_dir(challenges_dir)
    challenge_paths = _resolve_challenge_paths(challenges, base_dir)

    dataset = create_dataset(challenge_paths)

    if variants is not None:
        variant_set = {variants} if isinstance(variants, str) else set(variants)
        dataset = filter_dataset_by_variant(dataset, variant_set)

    return Task(
        dataset=dataset,
        solver=default_solver(max_messages=max_messages),
        scorer=check_store(STORE_KEY_ATTACK_SUCCESSFUL),
        name="cvebench",
    )


__all__ = ["cvebench"]
