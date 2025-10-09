#!/usr/bin/env python3
"""
Upload CTF Archive challenges to HuggingFace Dataset.

This script uploads the challenges directory to HuggingFace Hub so that
ctf_archive.py can download them on-demand instead of storing in git.

Usage:
    python upload_to_huggingface.py

Requirements:
    pip install huggingface_hub
    huggingface-cli login
"""

import logging
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPO_ID = "groq/ctf-archive-challenges"
REPO_TYPE = "dataset"
CHALLENGES_DIR = Path(__file__).parent / "challenges"


def main():
    """Upload challenges to HuggingFace."""
    if not CHALLENGES_DIR.exists():
        logger.error(f"Challenges directory not found: {CHALLENGES_DIR}")
        return

    logger.info(f"Uploading {CHALLENGES_DIR} to {REPO_ID}...")

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,
        )
        logger.info(f"Repository {REPO_ID} ready")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return

    # Upload the entire challenges directory
    try:
        api.upload_folder(
            folder_path=str(CHALLENGES_DIR),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo="challenges",
        )
        logger.info(f"✓ Successfully uploaded challenges to {REPO_ID}")
        logger.info(
            f"\nDataset available at: https://huggingface.co/datasets/{REPO_ID}"
        )
    except Exception as e:
        logger.error(f"Failed to upload: {e}")
        return


if __name__ == "__main__":
    main()
