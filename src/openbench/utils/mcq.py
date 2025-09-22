from typing import Any, List, Optional, Annotated
from pydantic import BeforeValidator
from inspect_ai.dataset import Sample, hf_dataset, csv_dataset, json_dataset, Dataset
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from inspect_ai import Task


# ----------- MCQ SAMPLE VALIDATION HELPERS -----------


def validate_input(value: Any) -> str:
    """Validate the input field of an MCQSample, must be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input must be a non-empty string")
    return value


def validate_choices(value: Any) -> List[str]:
    """Validate the choices field of an MCQSample, must be list of strings."""
    if not isinstance(value, list) or not value:
        raise ValueError("choices must be a non-empty list")
    if not all(isinstance(c, str) and c.strip() for c in value):
        raise ValueError("each choice must be a non-empty string")
    return value


def validate_target(value: Any) -> str:
    """Validate the target field of an MCQSample, must be a single uppercase letter."""
    if (
        not isinstance(value, str)
        or len(value) != 1
        or not value.isalpha()
        or not value.isupper()
    ):
        raise ValueError("target must be a single uppercase letter (e.g. 'A')")
    return value


# ----------- MCQ SAMPLE MODEL -----------


class MCQSample(Sample):
    """
    Minimal MCQ sample built on Inspect AI's `Sample`, with validators for MCQ fields.
    Users are expected to provide: record_to_mcq_sample(record) -> MCQSample.
    """

    input: Annotated[str, BeforeValidator(validate_input)]
    choices: Annotated[List[str], BeforeValidator(validate_choices)]
    target: Annotated[str, BeforeValidator(validate_target)]


# ----------- DATASET WRAPPER -----------
def make_dataset(
    dataset_type: str,
    dataset_path: str,
    record_to_mcq_sample,
    *,
    split: Optional[str] = None,
    auto_id: bool = True,
) -> Dataset:
    """
    Wrap Inspect AI dataset constructors.
    Supports records -> MCQSample -> Dataset.
    """

    if dataset_type == "hf":
        if split is None:
            raise ValueError("For dataset_type='hf', you must provide split")
        return hf_dataset(
            dataset_path,
            split=split,
            sample_fields=record_to_mcq_sample,
            auto_id=auto_id,
        )
    elif dataset_type == "csv":
        return csv_dataset(
            dataset_path, sample_fields=record_to_mcq_sample, auto_id=auto_id
        )
    elif dataset_type == "json":
        return json_dataset(
            dataset_path, sample_fields=record_to_mcq_sample, auto_id=auto_id
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


# ----------- TASK FACTORY -----------
def MCQEval(
    *,
    dataset_type: str,
    dataset_path: str,
    record_to_mcq_sample,
    split: Optional[str] = None,
    auto_id: bool = True,
) -> "Task":
    """
    Build a Task using a user-provided record_to_mcq_sample().

    Args:
        dataset_type: The type of dataset to load ('hf', 'csv', 'json')
        dataset_path: The path to the dataset
        record_to_mcq_sample: A function that converts a record to an MCQSample
        split: The split of the dataset to load (required for 'hf')
        auto_id: Whether to auto-generate an id for each sample

    Returns:
        Task: A Task object with the dataset, solver, and scorer
    """
    dataset = make_dataset(
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=auto_id,
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )
