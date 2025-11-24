"""ProgressiveMCPBench dataset loader.

ProgressiveMCPBench is a benchmark for evaluating LLM agents on real-world tasks
using the Model Context Protocol (MCP). It starts as a clone of LiveMCPBench
but uses a local JSON dataset and exact/fuzzy answer matching.
"""

import json
import logging
from pathlib import Path
from typing import Any, List

from inspect_ai.dataset import Dataset, Sample

logger = logging.getLogger(__name__)

# Store data relative to this file so it's editable by the user
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "progressivemcpbench.json"


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a ProgressiveMCPBench record to an Inspect Sample.

    Args:
        record: A dictionary containing ProgressiveMCPBench fields.

    Returns:
        Sample: Converted sample for evaluation.
    """
    answers = record.get("answers") or []
    if isinstance(answers, str):
        answers_list: List[str] = [answers.strip()] if answers.strip() else []
    else:
        answers_list = [str(a).strip() for a in answers if str(a).strip()]

    # Skip records with no usable answers (requirement 3: skip empty answers)
    # if not answers_list:
    #    raise ValueError("Empty answers list; record should be filtered before Sample creation")

    return Sample(
        id=record["task_id"],
        input=record["Question"],
        target=answers_list,  # list of acceptable answers
        metadata={
            "category": record.get("category"),
            "file_name": record.get("file_name"),
            "annotator_metadata": record.get("Annotator Metadata", {}),
        },
    )


def get_dataset() -> Dataset:
    """Load ProgressiveMCPBench dataset from a local JSON file."""
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            raw_records: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read dataset file {DATA_FILE}: {e}")
        # If file is missing, we raise instead of downloading, as we now commit the dataset
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}. Please ensure it is present.") from e

    samples: list[Sample] = []
    for record in raw_records:
        samples.append(record_to_sample(record))

    return samples
