"""
Rootly Terraform Multiple Choice Questions
Authored by:
Rootly AI Labs

# run evaluation
uv run openbench eval rootly_terraform --model "groq/llama-3.1-8b-instant" --T subtask=azure-k8s-mcq

Available subtasks:
- azure-k8s-mcq
- s3-security-mcq
"""

from typing import Optional
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval
from openbench.datasets.rootly_terraform import (
    get_dataset_config,
    record_to_mcq_sample,
)


@task
def rootly_terraform(subtask: Optional[str] = None) -> Task:  # type: ignore
    """Rootly Terraform MCQ evaluation with optional subtask selection."""
    dataset_config = get_dataset_config(subtask)

    return MCQEval(
        name="rootly_terraform",
        dataset_path=dataset_config["path"],
        record_to_mcq_sample=record_to_mcq_sample,
        split="test",
        dataset_kwargs={"revision": dataset_config["revision"]},
        config=GenerateConfig(),
    )
