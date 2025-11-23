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
    if not answers_list:
        raise ValueError("Empty answers list; record should be filtered before Sample creation")

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


def _ensure_local_json_dataset() -> None:
    """Download ICIP/LiveMCPBench from Hugging Face and cache to local JSON."""
    if DATA_FILE.is_file():
        return

    from datasets import load_dataset  # lazy import to avoid heavy dependency if not needed

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ICIP/LiveMCPBench test split to %s", DATA_FILE)

    try:
        ds = load_dataset("ICIP/LiveMCPBench", split="test")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    records: List[dict[str, Any]] = []

    for rec in ds:
        answers = rec.get("answers") or []
        # Normalize to list and filter empties here so JSON file only has graded tasks
        if isinstance(answers, str):
            answers_list = [answers.strip()] if answers.strip() else []
        else:
            answers_list = [str(a).strip() for a in answers if str(a).strip()]

        if not answers_list:
            continue

        # Create a clean dict
        clean_rec = dict(rec)
        clean_rec["answers"] = answers_list
        records.append(clean_rec)

    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(records)} records to {DATA_FILE}")


def get_dataset() -> Dataset:
    """Load ProgressiveMCPBench dataset from a local JSON file.
    
    Automatically downloads the dataset if it doesn't exist.
    """
    _ensure_local_json_dataset()

    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            raw_records: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read dataset file {DATA_FILE}: {e}")
        raise

    samples: list[Sample] = []
    for record in raw_records:
        try:
            samples.append(record_to_sample(record))
        except ValueError:
            # Should be rare since we filtered earlier, but stay robust
            logger.warning("Skipping record with empty or invalid answers: %s", record.get("task_id"))

    return Dataset(samples=samples)
