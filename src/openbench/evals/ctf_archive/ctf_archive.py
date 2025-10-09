"""
CTF Archive - Historical CTF challenges from pwn.college's archive.

This module ports challenges from https://github.com/pwncollege/ctf-archive
into the inspect_cyber framework, making 261 real-world CTF challenges
available for AI agent evaluation.

Challenges are downloaded on-demand from HuggingFace and cached locally at:
~/.cache/openbench/ctf_archive/
"""

import logging
import shutil
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import AgentAttempts, react
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, python
from openbench.utils.text import (
    CYBENCH_CONTINUE_MESSAGE,
    CYBENCH_INCORRECT_MESSAGE,
    CYBENCH_SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)

# HuggingFace dataset configuration
HF_REPO_ID = "groq/ctf-archive-challenges"
HF_REPO_TYPE = "dataset"

# Cache directory for downloaded challenges
CACHE_DIR = Path.home() / ".cache" / "openbench" / "ctf_archive"


def _ensure_challenges_downloaded() -> Path:
    """
    Ensure CTF challenges are downloaded and cached locally.

    Downloads challenges from HuggingFace on first use and caches them.
    Subsequent runs use the cached version.

    Returns:
        Path to challenges directory (cached location)
    """
    challenges_dir = CACHE_DIR / "challenges"

    # Check if already cached
    if challenges_dir.exists() and any(challenges_dir.iterdir()):
        logger.debug(f"Using cached challenges from {challenges_dir}")
        return challenges_dir

    logger.info("Downloading CTF Archive challenges from HuggingFace (first run)...")
    logger.info(f"This may take a few minutes. Caching to: {challenges_dir}")

    try:
        from huggingface_hub import snapshot_download

        # Download the entire dataset to cache
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            cache_dir=CACHE_DIR / "hf_cache",
        )

        # Move challenges to expected location
        downloaded_challenges = Path(downloaded_path) / "challenges"
        if downloaded_challenges.exists():
            challenges_dir.parent.mkdir(parents=True, exist_ok=True)
            if challenges_dir.exists():
                shutil.rmtree(challenges_dir)
            shutil.move(str(downloaded_challenges), str(challenges_dir))
            logger.info(f"✓ Challenges cached at: {challenges_dir}")
        else:
            raise FileNotFoundError(
                f"Downloaded dataset missing 'challenges' directory at {downloaded_path}"
            )

    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download CTF Archive challenges.\n"
            "Install with: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download challenges from {HF_REPO_ID}: {e}\n"
            f"Please check your internet connection and try again."
        )

    return challenges_dir


@task
def ctf_archive(
    ctf_names: str | list[str] | None = None,
    categories: str | list[str] | None = None,
    variant_names: str | list[str] | None = "hard",
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

    Returns:
        Task: Inspect AI task configured for CTF Archive challenges with proper
            agent and sandbox setup via inspect_cyber.

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

    # Ensure challenges are downloaded and cached
    challenges_dir = _ensure_challenges_downloaded()

    # Load dataset using inspect_cyber's loader
    dataset = create_agentic_eval_dataset(root_dir=challenges_dir.absolute())

    # Filter by CTF name
    if ctf_names:
        dataset = dataset.filter_by_metadata_field("ctf", ctf_names)

    # Filter by category
    if categories:
        dataset = dataset.filter_by_metadata_field("category", categories)

    # Filter out challenges without known flags
    dataset = dataset.filter(lambda sample: sample.target != "unknown")

    # Filter by variant
    if variant_names:
        dataset = dataset.filter_by_metadata_field("variant_name", variant_names)

    logger.info(f"Loaded {len(dataset)} CTF Archive challenges")

    return Task(
        dataset=dataset,
        solver=react(
            prompt=CYBENCH_SYSTEM_MESSAGE,
            tools=[bash(timeout=180), python(timeout=180)],
            attempts=AgentAttempts(
                attempts=3,
                incorrect_message=CYBENCH_INCORRECT_MESSAGE,
            ),
            on_continue=CYBENCH_CONTINUE_MESSAGE,
        ),
        scorer=includes(),
    )
