"""ProgressiveMCPBench dataset loader.

ProgressiveMCPBench is a benchmark for evaluating LLM agents on real-world tasks
using the Model Context Protocol (MCP). It starts as a clone of LiveMCPBench
but uses a local JSON dataset and exact/fuzzy answer matching.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

logger = logging.getLogger(__name__)

# Store data relative to this file so it's editable by the user
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "progressivemcpbench.json"

# Synthetic MCP data directory (in repo root)
# Path: src/openbench/datasets/ -> repo root is 3 parents up
SYNTHETIC_MCP_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "synthetic_mcp"
)
SYNTHETIC_TASKS_FILE = (
    SYNTHETIC_MCP_DIR / "tasks" / "progressivemcpbench_synthetic.json"
)


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
    """Load ProgressiveMCPBench dataset from a local JSON file."""
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            raw_records: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read dataset file {DATA_FILE}: {e}")
        # If file is missing, we raise instead of downloading, as we now commit the dataset
        raise FileNotFoundError(
            f"Dataset file not found at {DATA_FILE}. Please ensure it is present."
        ) from e

    samples: list[Sample] = []
    for record in raw_records:
        sample = record_to_sample(record)
        if sample:
            samples.append(sample)

    return MemoryDataset(samples=samples, name="progressivemcpbench")


def get_synthetic_dataset() -> Dataset:
    """Load the synthetic ProgressiveMCPBench dataset.

    This dataset uses the synthetic MCP server for deterministic evaluation.
    It loads from synthetic_mcp/tasks/progressivemcpbench_synthetic.json.
    """
    try:
        with SYNTHETIC_TASKS_FILE.open("r", encoding="utf-8") as f:
            raw_records: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(
            f"Failed to read synthetic dataset file {SYNTHETIC_TASKS_FILE}: {e}"
        )
        raise FileNotFoundError(
            f"Synthetic dataset file not found at {SYNTHETIC_TASKS_FILE}. "
            "Run the synthetic generation pipeline first (see docs/evals/progressivemcpbench.mdx)."
        ) from e

    samples: list[Sample] = []
    for record in raw_records:
        sample = record_to_sample(record)
        if sample:
            samples.append(sample)

    return MemoryDataset(samples=samples, name="progressivemcpbench-synthetic")
