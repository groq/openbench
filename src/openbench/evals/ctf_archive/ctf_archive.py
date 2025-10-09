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
import sys
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

# CTF Archive source repository
CTF_ARCHIVE_REPO = "https://github.com/pwncollege/ctf-archive.git"

# Cache directory for downloaded challenges
CACHE_DIR = Path.home() / ".cache" / "openbench" / "ctf_archive"


def _ensure_challenges_downloaded() -> Path:
    """
    Ensure CTF challenges are available.

    Checks in order:
    1. Local challenges directory (for development)
    2. Cached challenges in ~/.cache/openbench/ctf_archive/
    3. Downloads from pwn.college/ctf-archive and caches

    Returns:
        Path to challenges directory
    """
    import subprocess

    # First check local directory (for development/testing)
    local_challenges = Path(__file__).parent / "challenges"
    if local_challenges.exists() and any(local_challenges.iterdir()):
        logger.debug(f"Using local challenges from {local_challenges}")
        return local_challenges

    # Check cache
    cached_challenges = CACHE_DIR / "challenges"
    if cached_challenges.exists() and any(cached_challenges.iterdir()):
        logger.debug(f"Using cached challenges from {cached_challenges}")
        return cached_challenges

    # Need to download
    logger.info("Downloading CTF Archive from pwn.college (first run)...")
    logger.info(f"This may take a few minutes. Caching to: {CACHE_DIR}")

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        repo_dir = CACHE_DIR / "ctf-archive-repo"

        # Clone if not already cloned
        if not repo_dir.exists():
            logger.info(f"Cloning {CTF_ARCHIVE_REPO}...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", CTF_ARCHIVE_REPO, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")

        # Copy CTFs with flags to cache
        cached_challenges.mkdir(parents=True, exist_ok=True)

        logger.info("Copying challenges with verified flags...")
        copied_count = 0
        for ctf_dir in sorted(repo_dir.iterdir()):
            if not ctf_dir.is_dir() or ctf_dir.name.startswith("."):
                continue
            # Only copy CTFs that have .flag.sha256 files
            if list(ctf_dir.rglob(".flag.sha256")):
                target = cached_challenges / ctf_dir.name
                if not target.exists():
                    shutil.copytree(ctf_dir, target)
                    copied_count += 1

        # Clean up cloned repo to save space
        shutil.rmtree(repo_dir)

        logger.info(f"✓ Cached {copied_count} CTFs at: {cached_challenges}")

        # Now run convert_metadata to generate eval.yaml files
        logger.info("Generating eval.yaml metadata...")
        convert_script = Path(__file__).parent / "convert_metadata.py"
        result = subprocess.run(
            [sys.executable, str(convert_script)],
            cwd=cached_challenges.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Metadata conversion had issues: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timeout while cloning {CTF_ARCHIVE_REPO}. "
            "Please check your internet connection and try again."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download challenges: {e}\n"
            f"Please check your internet connection and try again."
        )

    return cached_challenges


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
