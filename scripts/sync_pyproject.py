#!/usr/bin/env python
"""Sync the root pyproject.toml into packages/pyproject.toml without CLI entrypoints."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "pyproject.toml"
TARGET_FILE = PROJECT_ROOT / "packages" / "openbench-core" / "pyproject.toml"
ROOT_FROM_TARGET = Path(os.path.relpath(PROJECT_ROOT, TARGET_FILE.parent)).as_posix()


def _join_root(path: str) -> str:
    if ROOT_FROM_TARGET in ("", "."):
        return path
    if path in (".", "./"):
        return ROOT_FROM_TARGET
    return f"{ROOT_FROM_TARGET}/{path}"


REMOVE_PROJECT_SCRIPTS = re.compile(
    r"^\[project\.scripts\]\n(?:^(?!\[).*\n?)+", flags=re.MULTILINE
)


Replacement = tuple[re.Pattern[str], Callable[[re.Match[str]], str]]


def adjust_relative_paths(content: str) -> str:
    """Update known relative paths so they remain valid from packages/openbench-core/."""
    replacements: tuple[Replacement, ...] = (
        (
            re.compile(r'(readme\s*=\s*)"README\.md"'),
            lambda m: f'{m.group(1)}"{_join_root("README.md")}"',
        ),
        (
            re.compile(r'(package-dir\s*=\s*\{""\s*=\s*)"src"'),
            lambda m: f'{m.group(1)}"{_join_root("src")}"',
        ),
        (
            re.compile(r'(where\s*=\s*\[)"src"'),
            lambda m: f'{m.group(1)}"{_join_root("src")}"',
        ),
        (
            re.compile(r'(testpaths\s*=\s*\[)"tests"'),
            lambda m: f'{m.group(1)}"{_join_root("tests")}"',
        ),
        (
            re.compile(r'(pythonpath\s*=\s*\[)"src",\s*"\."'),
            lambda m: f'{m.group(1)}"{_join_root("src")}", "{_join_root(".")}"',
        ),
        (
            re.compile(r'(source\s*=\s*\[)"src/openbench"'),
            lambda m: f'{m.group(1)}"{_join_root("src/openbench")}"',
        ),
        (
            re.compile(r'(omit\s*=\s*\[)"tests/\*"'),
            lambda m: f'{m.group(1)}"{_join_root("tests/*")}"',
        ),
    )
    adjusted = content
    for pattern, replacement in replacements:
        adjusted, _ = pattern.subn(replacement, adjusted)

    if ROOT_FROM_TARGET not in ("", "."):
        adjusted = re.sub(
            r'"(src/[^"]*)"',
            lambda m: f'"{_join_root(m.group(1))}"',
            adjusted,
        )
        adjusted = re.sub(
            r'"(tests/[^"]*)"',
            lambda m: f'"{_join_root(m.group(1))}"',
            adjusted,
        )
    return adjusted


def remove_cli_entrypoints(content: str) -> str:
    """Strip the [project.scripts] section from the pyproject."""
    return REMOVE_PROJECT_SCRIPTS.sub("", content)


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source pyproject not found at {SOURCE_FILE}")

    content = SOURCE_FILE.read_text(encoding="utf-8")
    content = remove_cli_entrypoints(content)
    content = adjust_relative_paths(content)
    # Collapse double blank lines introduced by the removal step.
    content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"

    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
