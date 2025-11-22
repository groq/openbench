"""Dataset loader for the Anthropic Political Even-handedness benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset


@dataclass
class PoliticalPromptPair:
    """Structured representation of a political prompt pair."""

    main_category: str
    topic_name: str
    partisan: bool
    template_category: str
    template: str
    stance_a: str
    stance_b: str
    prompt_a: str
    prompt_b: str
    prompt_a_group: str
    prompt_b_group: str
    pair_id: str

    def to_metadata(self) -> Dict[str, str | bool]:
        return {
            "main_category": self.main_category,
            "topic_name": self.topic_name,
            "partisan": self.partisan,
            "template_category": self.template_category,
            "template": self.template,
            "stance_a": self.stance_a,
            "stance_b": self.stance_b,
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "prompt_a_group": self.prompt_a_group,
            "prompt_b_group": self.prompt_b_group,
            "pair_id": self.pair_id,
        }


def _normalize_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "1", "yes", "y"}
    return False


def _generate_unique_id(prompt_a: str, prompt_b: str, topic_name: str) -> str:
    """Generate a unique, deterministic ID based on prompt content."""
    content = f"{prompt_a}|{prompt_b}"
    hash_digest = hashlib.sha256(content.encode()).hexdigest()[:12]

    return f"{topic_name}:{hash_digest}"


def _build_pair(record: dict) -> PoliticalPromptPair:
    template_category = str(record.get("template_category", "")).strip()
    topic_name = str(record.get("topic_name", "unknown")).strip()
    stance_a = str(record.get("stance_a", "")).strip()
    stance_b = str(record.get("stance_b", "")).strip()
    prompt_a = str(record.get("prompt_a", "")).strip()
    prompt_b = str(record.get("prompt_b", "")).strip()

    pair_id = _generate_unique_id(prompt_a, prompt_b, topic_name)

    return PoliticalPromptPair(
        main_category=str(record.get("main_category", "")).strip(),
        topic_name=topic_name,
        partisan=_normalize_bool(record.get("partisan")),
        template_category=template_category,
        template=str(record.get("template", "")).strip(),
        stance_a=stance_a,
        stance_b=stance_b,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        prompt_a_group=str(record.get("prompt_a_group", "")).strip(),
        prompt_b_group=str(record.get("prompt_b_group", "")).strip(),
        pair_id=pair_id,
    )


def record_to_sample(record: dict) -> Sample:
    """Convert a raw record from HF datasets to an Inspect Sample."""
    pair = _build_pair(record)

    serialized_input = (
        f"Prompt A ({pair.prompt_a_group}): {pair.prompt_a}\n"
        f"Prompt B ({pair.prompt_b_group}): {pair.prompt_b}"
    )

    return Sample(
        input=serialized_input,
        target="",
        id=pair.pair_id,
        metadata=pair.to_metadata(),
    )


def load_dataset(
    main_category: Optional[str] = None,
    topic_name: Optional[str] = None,
) -> Dataset:
    """
    Load and optionally filter the political even-handedness dataset.

    Args:
        main_category: Filter by broad category
        topic_name: Filter by fine-grained topic
    """
    dataset = hf_dataset(
        path="lvogel123/political-bias",
        split="test",
        sample_fields=record_to_sample,
    )

    samples = list(dataset)
    if not samples:
        raise ValueError(
            "No samples found in lvogel123/political-bias. "
            "Verify the dataset upload completed successfully."
        )
    filtered: List[Sample] = []

    for sample in samples:
        metadata: Dict[str, str | bool] = sample.metadata or {}

        if main_category and metadata.get("main_category") != main_category:
            continue

        if topic_name and metadata.get("topic_name") != topic_name:
            continue

        filtered.append(sample)

    if not filtered:
        raise ValueError(
            "No samples remain after applying filters. "
            "Relax the template/topic/main_category/partisan filters and try again."
        )

    name_parts = ["political_evenhandedness"]
    if main_category:
        name_parts.append(main_category)
    dataset_name = "_".join(name_parts)

    return MemoryDataset(samples=filtered, name=dataset_name)


__all__ = [
    "load_dataset",
    "record_to_sample",
]
