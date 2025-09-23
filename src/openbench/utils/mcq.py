from typing import Any, List, Optional, Annotated
from inspect_ai.model import GenerateConfig
from pydantic import BeforeValidator
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai import Task, Epochs
from openbench.scorers.mcq import create_mcq_scorer


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
    """Validate the target field: one of 'A', 'B', 'C', or 'D'."""
    if (
        not isinstance(value, str)
        or len(value) != 1
        or value not in ("A", "B", "C", "D")
    ):
        raise ValueError("target must be one of 'A', 'B', 'C', or 'D'")
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


# ----------- TASK FACTORY -----------
def MCQEval(
    *,
    name: str,
    dataset_path: str,
    record_to_mcq_sample,
    split: str,
    auto_id: bool = True,
    group_keys: Optional[List[str]] = None,
    additional_metrics: Optional[List[Any]] = None,
    prompt_template: Optional[str] = None,
    config: Optional[GenerateConfig] = None,
    epochs: Optional[Epochs] = None,
) -> "Task":
    """
    Build a Task using a user-provided record_to_mcq_sample().

    Args:
        name: Task name.
        dataset_path: Hugging Face dataset path/name.
        record_to_mcq_sample: Function converting a raw record into an `MCQSample`.
        split: HF dataset split (e.g., "train", "validation", "test").
        auto_id: Auto-generate IDs for samples when true.
        group_keys: Optional metadata keys to group reported metrics by (e.g., ["category"], ["subject"]).
        additional_metrics: Optional additional metrics to include alongside accuracy/stderr/std.
        prompt_template: Optional system prompt prepended before `generate()`.
        config: Optional model `GenerateConfig` for this task (defaults to a new `GenerateConfig()`).
        epochs: Optional `Epochs` to repeat samples and reduce scores across repeats.

    Returns:
        Task: Configured Inspect AI task with dataset, solver, scorer, config, and epochs.
    """
    dataset = hf_dataset(
        dataset_path,
        split=split,
        sample_fields=record_to_mcq_sample,
        auto_id=auto_id,
    )

    solver = [generate()]
    if prompt_template:
        solver = [system_message(prompt_template), generate()]

    scorer = create_mcq_scorer(
        group_keys=group_keys,
        additional_metrics=additional_metrics,
    )()

    return Task(
        name=name,
        dataset=dataset,
        solver=solver,
        scorer=scorer,
        config=config if config else GenerateConfig(),
        epochs=epochs,
    )
