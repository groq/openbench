"""Dataset loader for DeepResearch Bench."""

from inspect_ai.dataset import Sample, Dataset, hf_dataset


def query_to_sample(query: dict) -> Sample:
    """Convert a DeepResearch Bench query to an Inspect Sample."""
    return Sample(
        input=query["prompt"],
        id=str(query["id"]),
        metadata={
            "task_id": query["id"],
            "topic": query["topic"],
            "language": query.get("language", "en"),
        },
    )


def get_query() -> Dataset:
    """
    Load DeepResearch Bench query dataset from the original data files (which has been uploaded to Hugging Face.)

    Returns:
        Dataset: The query dataset.
    """
    return hf_dataset(
        path="lee64/deepresearch-bench-query",
        sample_fields=query_to_sample,
        split="train",
    )


def criteria_to_sample(record: dict) -> Sample:
    """Convert a DeepResearch Bench criteria record to an Inspect Sample."""
    return Sample(
        input=record["prompt"],
        id=str(record["id"]),
        metadata={
            "task_id": record["id"],
            "prompt": record["prompt"],
            "dimension_weight": record["dimension_weight"],
            "criterions": record["criterions"],
        },
    )


def get_criteria() -> Dataset:
    """
    Load DeepResearch Bench criteria dataset from Hugging Face.

    Returns:
        Dataset: The criteria dataset.
    """
    return hf_dataset(
        path="lee64/deepresearch-bench-criteria",
        sample_fields=criteria_to_sample,
        split="train",
    )


def reference_to_sample(record: dict) -> Sample:
    """Convert a DeepResearch Bench reference record to an Inspect Sample."""
    return Sample(
        input=record["prompt"],
        target=record["article"],  # Store the reference article as target
        id=str(record["id"]),
        metadata={
            "task_id": record["id"],
            "prompt": record["prompt"],
            "article": record["article"],
        },
    )


def get_reference_raw() -> Dataset:
    """
    Load DeepResearch Bench raw reference dataset (with citations) from Hugging Face.
    Used for FACT evaluation which needs citations.

    Returns:
        Dataset: The raw reference dataset with citations.
    """
    return hf_dataset(
        path="lee64/deepresearch-bench-reference-raw",
        sample_fields=reference_to_sample,
        split="train",
    )


def get_reference_clean() -> Dataset:
    """
    Load DeepResearch Bench cleaned reference dataset (citations removed) from Hugging Face.
    Used for RACE evaluation which compares content quality.

    Returns:
        Dataset: The cleaned reference dataset without citations.
    """
    return hf_dataset(
        path="lee64/deepresearch-bench-reference-clean",
        sample_fields=reference_to_sample,
        split="train",
    )
