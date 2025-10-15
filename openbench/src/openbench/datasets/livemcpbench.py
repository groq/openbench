"""LiveMCPBench dataset loader.

LiveMCPBench is a benchmark for evaluating LLM agents on real-world tasks
using the Model Context Protocol (MCP). The dataset contains 95 tasks across
different categories like Finance, with each task including questions, answers,
category information, and annotator metadata.
"""

import logging
from typing import Any
from inspect_ai.dataset import Dataset, Sample, hf_dataset

logger = logging.getLogger(__name__)


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a LiveMCPBench record to an Inspect Sample.

    Args:
        record: A dictionary containing LiveMCPBench fields:
            - task_id: Unique identifier for the task
            - Question: The question/task description
            - answers: Expected answer(s)
            - category: Task category (e.g., 'Finance')
            - file_name: Associated file name
            - Annotator Metadata: Additional metadata about the task

    Returns:
        Sample: Converted sample for evaluation
    """
    return Sample(
        id=record["task_id"],
        input=record["Question"],
        target=record["answers"],
        metadata={
            "category": record["category"],
            "file_name": record["file_name"],
            "annotator_metadata": record["Annotator Metadata"],
        },
    )


def get_dataset() -> Dataset:
    """Load the LiveMCPBench dataset from HuggingFace.

    Returns:
        Dataset: The LiveMCPBench dataset configured for evaluation
    """
    dataset = hf_dataset(
        path="ICIP/LiveMCPBench",
        split="test",
        sample_fields=record_to_sample,
    )

    return dataset
