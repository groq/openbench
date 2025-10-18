"""FActScore dataset loader.

This dataset prepares biography prompts for the FActScore benchmark using the
official prompt entity lists. The prompts mirror the setup described in the
paper ("Question: Tell me a bio of <entity>.").
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from openbench.utils.factscore_cache import (
    FACTSCORE_CACHE_ENV,
    prompt_entities_path,
    resolve_cache_root,
)

DEFAULT_PROMPT_TEMPLATE = "Question: Tell me a bio of {topic}."


@dataclass
class FactScoreDatasetConfig:
    """Configuration for constructing the FActScore dataset."""

    split: str = "labeled"
    num_examples: int | None = None
    shuffle: bool = False
    seed: int = 42
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    cache_root: str | None = None


def _load_topics(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _maybe_sample(topics: Sequence[str], cfg: FactScoreDatasetConfig) -> list[str]:
    topics = list(topics)
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(topics)

    if cfg.num_examples is not None:
        topics = topics[: cfg.num_examples]

    return topics


def build_samples(topics: Iterable[str], prompt_template: str) -> list[Sample]:
    samples: list[Sample] = []
    for topic in topics:
        prompt = prompt_template.format(topic=topic)
        samples.append(
            Sample(
                id=topic,
                input=prompt,
                metadata={
                    "topic": topic,
                    "prompt_template": prompt_template,
                    "cache_env_var": FACTSCORE_CACHE_ENV,
                },
            )
        )
    return samples


def get_dataset(
    split: str = "labeled",
    num_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    cache_root: str | None = None,
) -> Dataset:
    """Load the FActScore prompt dataset.

    Args:
        split: ``"labeled"`` (183 entities) or ``"unlabeled"`` (500 entities)
        num_examples: Optional cap on the number of topics (after shuffling)
        shuffle: Whether to shuffle the entities before sampling
        seed: Seed used when shuffling
        prompt_template: Template used to construct prompts. Must contain
            ``{topic}``.
        cache_root: Optional override for the cache root containing FActScore
            assets. When ``None`` the loader respects the
            ``OPENBENCH_FACTSCORE_CACHE`` environment variable.

    Returns:
        MemoryDataset backed by the selected prompt entities.
    """

    cfg = FactScoreDatasetConfig(
        split=split,
        num_examples=num_examples,
        shuffle=shuffle,
        seed=seed,
        prompt_template=prompt_template,
        cache_root=cache_root,
    )

    # Resolve cache root early for informative errors if missing
    cache_root_path = resolve_cache_root(cfg.cache_root)
    topics_file = prompt_entities_path(split=cfg.split, cache_root=cache_root_path)
    topics = _maybe_sample(_load_topics(topics_file), cfg)

    samples = build_samples(topics, cfg.prompt_template)

    return MemoryDataset(samples=samples, name=f"factscore_{cfg.split}")
