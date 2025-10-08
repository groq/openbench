"""
CTF Archive - Historical CTF challenges from pwn.college's archive.

This module ports challenges from https://github.com/pwncollege/ctf-archive
into the inspect_cyber framework, making ~790 real-world CTF challenges
available for AI agent evaluation.
"""

import logging
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import AgentAttempts, react
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, python
from openbench.utils.text import (
    CYBENCH_INCORRECT_MESSAGE,
    CYBENCH_SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)

CHALLENGES_DIR = Path(__file__).parent / "challenges"


@task
def ctf_archive(
    ctf_names: str | list[str] | None = None,
    categories: str | list[str] | None = None,
    variant_names: str | list[str] | None = "hard",
    sandbox_type: Literal["local"] = "local",
) -> Task:
    """
    Create a task for CTF Archive challenges.

    Args:
        ctf_names (str | list[str] | None): Filter by CTF name(s)
            (e.g., "hsctf2019", "picoctf2019"). If None, all CTFs included.
        categories (str | list[str] | None): Filter by category
            (e.g., "crypto", "pwn", "web"). If None, all categories included.
        variant_names (str | list[str] | None): Difficulty variant ("easy" or "hard").
            Default is "hard".
        sandbox_type (str): Sandbox type. Currently only "local" is supported.

    Returns:
        Task: Inspect AI task configured for CTF Archive challenges.

    Examples:
        # Run all HSCTF 2019 challenges
        ctf_archive(ctf_names="hsctf2019")

        # Run only crypto challenges across all CTFs
        ctf_archive(categories="crypto")

        # Run PWN challenges from specific CTFs
        ctf_archive(
            ctf_names=["hsctf2019", "picoctf2019"],
            categories="pwn"
        )
    """
    from inspect_cyber import create_agentic_eval_dataset

    # Load dataset from challenges directory
    dataset = create_agentic_eval_dataset(
        root_dir=CHALLENGES_DIR.absolute(),
    )

    # Filter out challenges without known flags
    dataset = dataset.filter(lambda sample: sample.target != "unknown")

    # Apply filters
    if ctf_names:
        # Filter by CTF name via metadata field
        # Normalize by removing spaces and converting to lowercase for comparison
        def normalize(s: str) -> str:
            return s.lower().replace(" ", "").replace("-", "")

        def filter_by_ctf(sample):
            sample_ctf_norm = normalize(sample.metadata.get("ctf", ""))
            if isinstance(ctf_names, str):
                return normalize(ctf_names) in sample_ctf_norm
            return any(normalize(name) in sample_ctf_norm for name in ctf_names)

        dataset = dataset.filter(filter_by_ctf)

    if categories:
        # Filter by category
        dataset = dataset.filter_by_metadata_field("categories", categories)

    if variant_names:
        # Filter by difficulty variant
        dataset = dataset.filter_by_metadata_field("variant_name", variant_names)

    logger.info(f"Loaded {len(dataset)} CTF Archive challenges")

    return Task(
        dataset=dataset,
        solver=react(
            prompt=CYBENCH_SYSTEM_MESSAGE,
            tools=[bash(timeout=180), python(timeout=180)],
            attempts=AgentAttempts(
                attempts=1,
                incorrect_message=CYBENCH_INCORRECT_MESSAGE,
            ),
        ),
        scorer=includes(),
    )
