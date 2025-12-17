"""ProgressiveMCPBench dataset loader.

ProgressiveMCPBench is a benchmark for evaluating LLM agents on real-world tasks
using the Model Context Protocol (MCP). It uses a synthetic MCP server for
deterministic evaluation with exact/fuzzy answer matching.
"""

import json
import logging
from typing import Any, Optional

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from openbench.utils.progressivemcpbench_cache import (
    get_progressivemcpbench_dataset_dir,
)

logger = logging.getLogger(__name__)


def record_to_sample(record: dict[str, Any]) -> Optional[Sample]:
    """Convert a ProgressiveMCPBench record to an Inspect Sample.

    Args:
        record: A dictionary containing ProgressiveMCPBench fields.

    Returns:
        Sample: Converted sample for evaluation.
        None: If the record should be skipped (e.g. answer is None or empty).
    """
    # specific user request: if answer is explicitly null, skip the task
    if record.get("answer") is None:
        return None

    # Handle answer as a single string
    raw_answer = record.get("answer")
    if isinstance(raw_answer, list):
        # If it's a list, take the first non-empty answer
        answer = next((str(a).strip() for a in raw_answer if str(a).strip()), "")
    else:
        answer = str(raw_answer).strip() if raw_answer else ""

    # Skip records with no usable answer
    if not answer:
        return None

    metadata = {
        "category": record.get("category"),
        "file_name": record.get("file_name"),
        "annotator_metadata": record.get("Annotator Metadata", {}),
    }

    # Add tool requirement annotations if present (for minimal strategies)
    if "required_servers" in record:
        metadata["required_servers"] = record["required_servers"]
    if "required_tools" in record:
        metadata["required_tools"] = record["required_tools"]

    # Add scorer instructions if present
    if record.get("scorer_instructions"):
        metadata["scorer_instructions"] = record["scorer_instructions"]

    return Sample(
        id=record["task_id"],
        input=record["Question"],
        target=answer,  # single answer string
        metadata=metadata,
    )


def get_dataset() -> Dataset:
    """Load ProgressiveMCPBench dataset.

    This dataset uses the synthetic MCP server for deterministic evaluation.
    It loads from the progressivemcpbench sibling repository.
    """
    dataset_dir = get_progressivemcpbench_dataset_dir()
    tasks_file = dataset_dir / "tasks" / "progressivemcpbench.json"
    try:
        with tasks_file.open("r", encoding="utf-8") as f:
            raw_records: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read dataset file {tasks_file}: {e}")
        raise FileNotFoundError(
            f"Dataset file not found at {tasks_file}. "
            "Ensure the progressivemcpbench repository is cloned as a sibling directory."
        ) from e

    samples: list[Sample] = []
    for record in raw_records:
        sample = record_to_sample(record)
        if sample:
            samples.append(sample)

    return MemoryDataset(samples=samples, name="progressivemcpbench")
