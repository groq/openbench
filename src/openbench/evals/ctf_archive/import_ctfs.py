#!/usr/bin/env python3
"""
Import CTFs from CTF Archive into openbench challenges directory.

Usage:
    python import_ctfs.py                    # Import Tier 1 + Tier 2 CTFs
    python import_ctfs.py --all              # Import all CTFs with flags
    python import_ctfs.py picoctf2019        # Import specific CTF
"""

import shutil
import sys
from pathlib import Path


def import_ctf(ctf_dir: Path, challenges_dir: Path) -> int:
    """
    Copy a CTF from CTF Archive to challenges directory.

    Args:
        ctf_dir: Source CTF directory in ctf-archive
        challenges_dir: Target challenges directory

    Returns:
        Number of challenges copied
    """
    target_dir = challenges_dir / ctf_dir.name

    # Skip if already exists
    if target_dir.exists():
        print(f"  Skipping {ctf_dir.name} (already exists)")
        return 0

    # Copy the entire CTF directory
    print(f"  Copying {ctf_dir.name}...")
    shutil.copytree(ctf_dir, target_dir)

    # Count challenges with flags
    flag_count = len(list(target_dir.rglob(".flag.sha256")))
    print(f"  ✓ Copied {ctf_dir.name} ({flag_count} challenges with flags)")

    return flag_count


def main():
    # Source CTF Archive directory
    # Go up from src/openbench/evals/ctf_archive to groq/ directory
    ctf_archive_root = (
        Path(__file__).parent.parent.parent.parent.parent.parent / "ctf-archive"
    )

    # Target challenges directory
    challenges_dir = Path(__file__).parent / "challenges"
    challenges_dir.mkdir(exist_ok=True)

    # CTF directory names to import
    # Tier 1: High Priority (10+ flags)
    tier1_ctfs = [
        "picoctf2019",
        "csawctf2011",
        "patriotctf2022",
        "angstromctf2018",
        "hsctf2019",
        "hitcon2017quals",
        "byuctf2023",
        "accessdeniedctf2022",
    ]

    # Tier 2: Medium-High Priority (7-9 flags)
    tier2_ctfs = [
        "hsctf2020",
        "angstromctf2019",
        "0x41414141ctf2021",
        "imaginaryctf2023",
        "csawctf2014",
        "neverlan2019",
        "downunderctf2024",
    ]

    # Determine which CTFs to import
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Import all CTFs that have any flags
            ctfs_to_import = []
            for ctf_dir in sorted(ctf_archive_root.iterdir()):
                if ctf_dir.is_dir() and list(ctf_dir.rglob(".flag.sha256")):
                    ctfs_to_import.append(ctf_dir.name)
            print(f"Importing ALL {len(ctfs_to_import)} CTFs with flags...\n")
        else:
            # Import specific CTF
            ctfs_to_import = [sys.argv[1]]
            print(f"Importing {sys.argv[1]}...\n")
    else:
        # Default: import Tier 1 + Tier 2
        ctfs_to_import = tier1_ctfs + tier2_ctfs
        print(f"Importing Tier 1 + Tier 2 CTFs ({len(ctfs_to_import)} total)...\n")

    # Import each CTF
    total_challenges = 0
    imported_ctfs = 0

    for ctf_name in ctfs_to_import:
        ctf_path = ctf_archive_root / ctf_name
        if not ctf_path.exists():
            print(f"Warning: {ctf_path} does not exist, skipping...")
            continue

        count = import_ctf(ctf_path, challenges_dir)
        if count > 0:
            total_challenges += count
            imported_ctfs += 1

    print(f"\nImported {imported_ctfs} CTFs with {total_challenges} total challenges")
    print("\nNext step: Run convert_metadata.py to generate eval.yaml files")


if __name__ == "__main__":
    main()
