#!/usr/bin/env python3
"""
Export CHANGELOG.md to docs/changelog.mdx with frontmatter.

This script reads the root CHANGELOG.md file and generates the docs/changelog.mdx
file with appropriate MDX frontmatter for the documentation site.

Output (fixed path):
- docs/changelog.mdx
"""

from __future__ import annotations
from pathlib import Path


def export_changelog(
    source_path: Path,
    output_path: Path,
    title: str = "Changelog",
    description: str = "Version history and release notes for openbench",
) -> None:
    """
    Read CHANGELOG.md and write to changelog.mdx with frontmatter.

    Args:
        source_path: Path to CHANGELOG.md
        output_path: Path to output changelog.mdx
        title: Page title for MDX frontmatter
        description: Page description for MDX frontmatter
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source changelog not found: {source_path}")

    # Read the changelog content
    changelog_content = source_path.read_text(encoding="utf-8")

    # Strip "# Changelog" header if present (we'll add it via frontmatter)
    lines = changelog_content.splitlines()
    if lines and lines[0].strip() == "# Changelog":
        changelog_content = "\n".join(lines[1:]).lstrip()

    # Construct MDX frontmatter
    frontmatter = f"""---
title: {title}
description: {description}
---

"""

    # Write the MDX file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        frontmatter + changelog_content,
        encoding="utf-8",
    )


def main() -> None:
    """Export changelog to MDX file under openbench/docs/"""
    source = Path("CHANGELOG.md")
    output = Path("docs/changelog.mdx")
    export_changelog(source_path=source, output_path=output)
    print(f"✓ Exported {source} → {output}")


if __name__ == "__main__":
    main()
