"""
Dataset loader adapted from CVE-bench: https://github.com/uiuc-kang-lab/cve-bench
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
import tempfile

import yaml
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from pydantic import BaseModel

CHALLENGE_INFO_FILENAME = "challenge.yml"
DEFAULT_ENV_VAR = "CVEBENCH_CHALLENGE_DIR"


class Variant(BaseModel, frozen=True):
    """Represents an individual CVEBench challenge variant."""

    prompt: str
    files: dict[str, str] = {}
    metadata: dict[str, object] = {}


class ChallengeInfo(BaseModel, frozen=True):
    """Parsed representation of a challenge.yml manifest."""

    name: str
    files: dict[str, str] = {}
    variants: dict[str, Variant]
    metadata: dict[str, object] = {}


def _default_challenges_dir() -> Path:
    """Resolve the packaged challenges directory."""

    package_root = Path(__file__).resolve().parent.parent
    return package_root / "evals" / "cvebench" / "challenges"


def resolve_challenges_dir(path: str | Path | None = None) -> Path:
    """Compute the directory that houses CVEBench challenge definitions."""

    if path is not None:
        return Path(path).resolve()

    env_override = os.environ.get(DEFAULT_ENV_VAR)
    if env_override:
        return Path(env_override).resolve()

    return _default_challenges_dir().resolve()


def create_dataset(challenge_dirs: list[Path]) -> Dataset:
    """Create an Inspect dataset from one or more challenge directories."""

    challenge_dirs = list(_find_challenge_dirs_recursive(challenge_dirs))
    if not challenge_dirs:
        raise ValueError("No valid challenges found in the provided directories")
    samples = list(_create_samples(challenge_dirs))
    return MemoryDataset(samples=samples, name="cvebench")


def filter_dataset_by_variant(dataset: Dataset, variants: set[str]) -> Dataset:
    """Filter dataset samples down to specific variant names."""

    return dataset.filter(
        lambda sample: sample.metadata is not None
        and sample.metadata.get("variant") in variants
    )


def _find_challenge_dirs_recursive(
    challenges_dirs: list[Path],
) -> Generator[Path, None, None]:
    for challenge_dir in challenges_dirs:
        if not challenge_dir.exists():
            raise NotADirectoryError(f"Could not find directory: '{challenge_dir}'")
        if not any(challenge_dir.iterdir()):
            raise NotADirectoryError(f"Challenge directory is empty: '{challenge_dir}'")
        for file in challenge_dir.rglob(CHALLENGE_INFO_FILENAME):
            yield file.parent


def _create_samples(challenge_dirs: list[Path]) -> Generator[Sample, None, None]:
    sandbox_config_file = "compose.yml"

    for challenge_dir in challenge_dirs:
        challenge_info = _load_challenge_info(challenge_dir)

        challenge_files: dict[str, str] = _gather_challenge_tree(challenge_dir)
        challenge_files.update(
            _make_paths_absolute(challenge_info.files, challenge_dir)
        )

        for variant_name, variant in challenge_info.variants.items():
            variant_files = _make_paths_absolute(variant.files, challenge_dir)
            sandbox_config = str((challenge_dir / sandbox_config_file).resolve())

            env_file_path = _create_env_file(challenge_info.name, challenge_dir)

            files = challenge_files.copy()
            files.update(variant_files)
            files[".env"] = env_file_path

            yield Sample(
                id=f"{challenge_info.name}-{variant_name}",
                input=variant.prompt,
                files=files,
                metadata={
                    "variant": variant_name,
                    "challenge_metadata": challenge_info.metadata,
                    "variant_metadata": variant.metadata,
                },
                sandbox=("docker", sandbox_config),
            )


def _load_challenge_info(challenge: Path) -> ChallengeInfo:
    yaml_path = challenge / CHALLENGE_INFO_FILENAME
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            if not data:
                raise ValueError(f"Empty YAML file: '{yaml_path}'")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {CHALLENGE_INFO_FILENAME} in '{challenge}'."
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML format in '{yaml_path}': {exc}") from exc

    try:
        return ChallengeInfo(**data)
    except TypeError as exc:
        raise ValueError(
            f"Invalid challenge info format in '{yaml_path}': {exc}"
        ) from exc


def _make_paths_absolute(files: dict[str, str], base_path: Path) -> dict[str, str]:
    return {
        destination: _make_path_absolute(source, base_path)
        for destination, source in files.items()
    }


def _make_path_absolute(path_or_content: str, base_path: Path) -> str:
    potential = Path(path_or_content)
    if potential.is_absolute():
        return str(potential)
    candidate = base_path / path_or_content
    if candidate.is_file():
        return str(candidate.resolve())
    return path_or_content


def _gather_challenge_tree(challenge_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for file_path in challenge_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(challenge_dir).as_posix()
            mapping[relative] = str(file_path)
    return mapping


def _create_env_file(challenge_name: str, challenge_dir: Path) -> str:
    original_env = challenge_dir / ".env"
    contents = ""
    if original_env.exists():
        contents = original_env.read_text(encoding="utf-8")
        if not contents.endswith("\n"):
            contents += "\n"

    challenge_root = challenge_dir.parent.resolve()
    lines = {
        "CVE": challenge_name,
        "CVE_LOWER": challenge_name.lower(),
        "CVEBENCH_TARGET_DIR": str((challenge_dir / "target").resolve()),
        "CVEBENCH_CHALLENGE_DIR": str(challenge_root),
    }

    for key, value in lines.items():
        contents += f"{key}={value}\n"

    tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    return tmp.name


__all__ = [
    "Variant",
    "ChallengeInfo",
    "create_dataset",
    "filter_dataset_by_variant",
    "resolve_challenges_dir",
]
