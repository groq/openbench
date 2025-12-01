#!/usr/bin/env python3
"""
Validate and auto-fix "mcq" tags for MCQ benchmarks.

This script uses the centralized is_mcq_task() function to detect MCQ benchmarks
by inspecting their actual implementations, then ensures they all have the
"mcq" tag in config.py.

Performance Note:
    This script loads all benchmarks (~200+) to check their scorers via
    is_mcq_task(). This task is implemented in CI/CD since it takes 2-3 minutes 
    and runs infrequently (only when config.py or eval files change in PRs).
    The detection is comprehensive and catches all inconsistencies.

Usage:
    # Check for missing or incorrect tags
    python3 scripts/validate_mcq_tags.py

    # Auto-fix config.py (adds missing tags, removes incorrect ones)
    python3 scripts/validate_mcq_tags.py --fix

    # Only check alpha benchmarks
    python3 scripts/validate_mcq_tags.py --alpha-only
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openbench.config import get_all_benchmarks, BENCHMARKS
from openbench.utils.mcq import get_mcq_benchmarks


def validate_mcq_tags(include_alpha: bool = True) -> Tuple[List[str], List[str]]:
    """
    Validate that MCQ benchmarks have the "mcq" tag.

    Args:
        include_alpha: Whether to check alpha benchmarks

    Returns:
        Tuple of (missing_tag_benchmarks, incorrect_tag_benchmarks)
        - missing_tag: MCQ benchmarks without "mcq" tag
        - incorrect_tag: Non-MCQ benchmarks with "mcq" tag
    """
    print("🔍 Detecting MCQ benchmarks by inspecting implementations...")
    mcq_benchmarks = set(get_mcq_benchmarks(include_alpha=include_alpha))
    print(f"   Found {len(mcq_benchmarks)} MCQ benchmarks")

    print("\n🔍 Checking tags in config.py...")
    all_benchmarks = get_all_benchmarks(include_alpha=include_alpha)

    missing_tag = []
    incorrect_tag = []

    for benchmark_name, metadata in all_benchmarks.items():
        is_mcq = benchmark_name in mcq_benchmarks
        has_tag = "mcq" in metadata.tags

        if is_mcq and not has_tag:
            missing_tag.append(benchmark_name)
        elif not is_mcq and has_tag:
            incorrect_tag.append(benchmark_name)

    return missing_tag, incorrect_tag


def fix_config_file(missing_tag: List[str], incorrect_tag: List[str]) -> None:
    """
    Update config.py to fix missing and incorrect tags.

    Args:
        missing_tag: List of benchmarks missing "mcq" tag
        incorrect_tag: List of benchmarks incorrectly tagged
    """
    config_path = Path(__file__).parent.parent / "src" / "openbench" / "config.py"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    print(f"\n📝 Reading {config_path}...")
    content = config_path.read_text()
    original_content = content

    # Get current tags for each benchmark
    all_benchmarks = get_all_benchmarks(include_alpha=True)

    # Add missing "mcq" tags
    for benchmark in missing_tag:
        print(f"   ➕ Adding 'mcq' tag to {benchmark}")

        if benchmark not in all_benchmarks:
            print(f"      ⚠️  Benchmark not found: {benchmark}")
            continue

        current_tags = list(all_benchmarks[benchmark].tags)  # Copy the list

        # Skip if already has "mcq" (shouldn't happen, but be safe)
        if "mcq" in current_tags:
            continue

        # Add "mcq" at the beginning
        new_tags = current_tags.copy()
        new_tags.insert(0, "mcq")

        # Build the old and new tags strings
        old_tags_str = 'tags=[' + ', '.join(f'"{t}"' for t in current_tags) + ']'
        new_tags_str = 'tags=[' + ', '.join(f'"{t}"' for t in new_tags) + ']'

        # Replace in content
        if old_tags_str in content:
            content = content.replace(old_tags_str, new_tags_str, 1)
        else:
            print(f"      ⚠️  Could not find exact tags match for {benchmark}")

    # Remove incorrect "mcq" tags
    for benchmark in incorrect_tag:
        print(f"   ➖ Removing 'mcq' tag from {benchmark}")

        if benchmark not in all_benchmarks:
            print(f"      ⚠️  Benchmark not found: {benchmark}")
            continue

        current_tags = list(all_benchmarks[benchmark].tags)  # Copy the list

        # Skip if doesn't have "mcq" (shouldn't happen, but be safe)
        if "mcq" not in current_tags:
            continue

        # Remove "mcq" from tags
        new_tags = current_tags.copy()
        new_tags.remove("mcq")

        # Build the old and new tags strings
        old_tags_str = 'tags=[' + ', '.join(f'"{t}"' for t in current_tags) + ']'
        new_tags_str = 'tags=[' + ', '.join(f'"{t}"' for t in new_tags) + ']'

        # Replace in content
        if old_tags_str in content:
            content = content.replace(old_tags_str, new_tags_str, 1)
        else:
            print(f"      ⚠️  Could not find exact tags match for {benchmark}")

    if content != original_content:
        print(f"\n💾 Writing changes to {config_path}...")
        config_path.write_text(content)
        print("✅ Config file updated successfully!")
    else:
        print("\n✅ No changes needed - config file is already correct")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and auto-fix 'mcq' tags for MCQ benchmarks"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix config.py by adding/removing tags",
    )
    parser.add_argument(
        "--alpha-only",
        action="store_true",
        help="Only check alpha/experimental benchmarks",
    )
    parser.add_argument(
        "--no-alpha",
        action="store_true",
        help="Exclude alpha/experimental benchmarks",
    )

    args = parser.parse_args()

    # Determine alpha inclusion
    if args.alpha_only and args.no_alpha:
        print("❌ Cannot use both --alpha-only and --no-alpha")
        sys.exit(1)

    include_alpha = not args.no_alpha

    print("=" * 70)
    print("MCQ Tag Validation")
    print("=" * 70)
    print(f"Checking: {'All benchmarks' if include_alpha else 'Non-alpha benchmarks only'}")
    print()

    try:
        missing_tag, incorrect_tag = validate_mcq_tags(include_alpha=include_alpha)

        # Filter by alpha-only if requested
        if args.alpha_only:
            all_benchmarks = get_all_benchmarks(include_alpha=True)
            missing_tag = [b for b in missing_tag if all_benchmarks[b].is_alpha]
            incorrect_tag = [b for b in incorrect_tag if all_benchmarks[b].is_alpha]

        # Report results
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)

        if missing_tag:
            print(f"\n❌ Missing 'mcq' tag ({len(missing_tag)} benchmarks):")
            for benchmark in sorted(missing_tag)[:10]:
                print(f"   • {benchmark}")
            if len(missing_tag) > 10:
                print(f"   ... and {len(missing_tag) - 10} more")

        if incorrect_tag:
            print(
                f"\n❌ Incorrect 'mcq' tag ({len(incorrect_tag)} benchmarks):"
            )
            print("   (These are not MCQ benchmarks but have the tag)")
            for benchmark in sorted(incorrect_tag)[:10]:
                print(f"   • {benchmark}")
            if len(incorrect_tag) > 10:
                print(f"   ... and {len(incorrect_tag) - 10} more")

        if not missing_tag and not incorrect_tag:
            print("\n✅ All MCQ tags are correct!")
            sys.exit(0)

        # Suggest fix
        if not args.fix:
            print("\n" + "-" * 70)
            print("To automatically fix these issues, run:")
            print(f"  python {Path(__file__).name} --fix")
            print("-" * 70)
            sys.exit(1)

        # Apply fixes
        print("\n" + "=" * 70)
        print("Applying Fixes")
        print("=" * 70)
        fix_config_file(missing_tag, incorrect_tag)

        # Verify fixes
        print("\n" + "=" * 70)
        print("Verifying Fixes")
        print("=" * 70)

        # Reload the config module to get updated tags
        import importlib
        import openbench.config
        importlib.reload(openbench.config)

        new_missing, new_incorrect = validate_mcq_tags(include_alpha=include_alpha)

        if args.alpha_only:
            all_benchmarks = get_all_benchmarks(include_alpha=True)
            new_missing = [b for b in new_missing if all_benchmarks[b].is_alpha]
            new_incorrect = [b for b in new_incorrect if all_benchmarks[b].is_alpha]

        if new_missing or new_incorrect:
            print("⚠️  Some issues remain after fixes:")
            if new_missing:
                print(f"   Still missing: {len(new_missing)}")
            if new_incorrect:
                print(f"   Still incorrect: {len(new_incorrect)}")
            print("\nYou may need to manually review config.py")
            sys.exit(1)
        else:
            print("✅ All issues fixed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
