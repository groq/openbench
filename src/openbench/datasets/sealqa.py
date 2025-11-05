from datetime import date, datetime

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset

_VALID_SUBSETS = {"seal_hard", "seal_0", "longseal"}

# Known dataset revision aliases (ISO date format). Add new dated snapshots here.
_REVISION_ALIASES = {
    "2025-10-29": "99f54adaf38bbc23e65fb218b0192cff3625fa76",
    "2025-10-15": "7b873f534d19651955e0486e1dc8778e375be3c9",
    "2025-09-29": "0db0c783915cba6bbcd0542af0750392ff0d3f18",
}


def record_to_sample(record: dict) -> Sample:
    """Convert a SealQA record to an Inspect Sample."""
    metadata = {k: v for k, v in record.items() if k not in {"question", "answer"}}
    sample_id = metadata.get("id")

    return Sample(
        input=record["question"],
        target=record["answer"],
        id=sample_id,
        metadata=metadata,
    )


def get_dataset(
    subset: str = "seal_hard",
    last_updated: str | date | datetime | None = None,
) -> Dataset:
    """Load a SealQA subset from Hugging Face.

    Args:
        subset: Which subset to load. One of "seal_hard", "seal_0", or "longseal".
        last_updated: Commit SHA, tag, or ISO date alias (e.g. "2025-10-29"). Use
            "latest" for the current HEAD snapshot.

    Returns:
        A `MemoryDataset` containing samples from the requested subset.
    """

    if subset not in _VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset '{subset}'. Valid options are: {sorted(_VALID_SUBSETS)}"
        )

    if last_updated is None:
        raise ValueError(
            "`last_updated` is required for sealqa datasets because snapshots are"
            " keyed by update date. Provide 'latest' or an update date such as"
            " '2025-10-29'."
        )

    if isinstance(last_updated, (date, datetime)):
        last_updated_str = last_updated.isoformat()
    else:
        last_updated_str = str(last_updated)

    normalized_last_updated = (
        last_updated_str.lower().replace(" ", "-").replace("/", "-")
    )
    resolved_revision = _REVISION_ALIASES.get(
        normalized_last_updated, normalized_last_updated
    )

    if resolved_revision != "latest":
        dataset = hf_dataset(
            path="vtllms/sealqa",
            name=subset,
            split="test",
            sample_fields=record_to_sample,
            revision=resolved_revision,
        )
    else:
        dataset = hf_dataset(
            path="vtllms/sealqa",
            name=subset,
            split="test",
            sample_fields=record_to_sample,
        )

    samples = list(dataset)

    if normalized_last_updated == "latest":
        warning_text = (
            "⚠️ This eval is using the latest SealQA dataset. "
            "SealQA refreshes over time, so future runs without a pinned `last_updated` "
            "date may use a different snapshot."
        )
        print(f"\033[1;33m{warning_text}\033[0m")

    dataset_name_suffix = (
        normalized_last_updated if normalized_last_updated != "latest" else "latest"
    )
    dataset_name = f"sealqa_{subset}_test_{dataset_name_suffix}"

    return MemoryDataset(samples=samples, name=dataset_name)
