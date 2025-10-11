from typing import Any, List, Optional, Annotated
from inspect_ai.model import (
    GenerateConfig,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessage,
)
from pydantic import BeforeValidator
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from inspect_ai import Task, Epochs
from openbench.scorers.mcq import create_mcq_scorer


# ----------- MCQ SAMPLE VALIDATION HELPERS -----------


def validate_input(value: Any) -> str | list[ChatMessage]:
    """Validate the input field of an MCQSample, must be a non-empty string or list of ChatMessage."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("input must be a non-empty string")
        return value
    elif isinstance(value, list):
        # Check if it's a list of ChatMessage-like objects
        chat_types = (
            ChatMessageSystem,
            ChatMessageUser,
            ChatMessageAssistant,
            ChatMessageTool,
        )
        if all(isinstance(item, chat_types) for item in value):
            return value
        else:
            raise ValueError(
                "input must be a non-empty string or list of ChatMessage objects"
            )
    else:
        raise ValueError(
            "input must be a non-empty string or list of ChatMessage objects"
        )


def validate_target(value: Any) -> str:
    """Validate the target field: must be single uppercase letter."""
    if not (isinstance(value, str) and len(value) == 1 and value.isupper()):
        raise ValueError("target must be a single uppercase letter.")
    return value


# ----------- MCQ SAMPLE MODEL -----------


class MCQSample(Sample):
    """
    Minimal MCQ sample built on Inspect AI's `Sample`, with validators for MCQ fields.
    Users are expected to provide: record_to_mcq_sample(record) -> MCQSample.
    """

    input: Annotated[str | list[ChatMessage], BeforeValidator(validate_input)]
    target: Annotated[str, BeforeValidator(validate_target)]


# ----------- TASK FACTORY -----------
def MCQEval(
    *,
    name: str,
    dataset_type: str = "hf",
    dataset_path: str,
    record_to_mcq_sample,
    split: Optional[str] = None,
    auto_id: bool = True,
    subset_name: Optional[str] = None,
    group_keys: Optional[List[str]] = None,
    additional_metrics: Optional[List[Any]] = None,
    prompt_template: Optional[str] = None,
    config: Optional[GenerateConfig] = None,
    epochs: Optional[Epochs] = None,
    dataset_kwargs: Optional[dict[str, Any]] = None,
) -> "Task":
    """
    Build a Task using a user-provided record_to_mcq_sample().

    Args:
        name: Task name.
        dataset_type: Dataset type (hf, csv, json).
        dataset_path: Dataset path/name.
        record_to_mcq_sample: Function converting a raw record into an `MCQSample`.
        split: HF dataset split (e.g., "train", "validation", "test").
        auto_id: Auto-generate IDs for samples when true.
        subset_name: Dataset subset name.
        group_keys: Optional metadata keys to group reported metrics by (e.g., ["category"], ["subject"]).
        additional_metrics: Optional additional metrics to include alongside accuracy/stderr/std.
        prompt_template: Optional system prompt prepended before `generate()`.
        config: Optional model `GenerateConfig` for this task (defaults to a new `GenerateConfig()`).
        epochs: Optional `Epochs` to repeat samples and reduce scores across repeats.
        dataset_kwargs: Optional additional dataset-specific parameters.

    Returns:
        Task: Configured Inspect AI task with dataset, solver, scorer, config, and epochs.
    """
    # Load the raw dataset first to enable filtering
    if dataset_type == "hf":
        if split is None:
            raise ValueError("For dataset_type='hf', you must provide split")
        from datasets import load_dataset as hf_load_dataset  # type: ignore

        raw_dataset = hf_load_dataset(
            dataset_path, name=subset_name, split=split, trust_remote_code=True
        )
    elif dataset_type == "csv":
        import pandas as pd  # type: ignore

        df = pd.read_csv(dataset_path)
        raw_dataset = df.to_dict("records")
    elif dataset_type == "json":
        import json as json_lib

        with open(dataset_path) as f:
            raw_dataset = json_lib.load(f)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Filter and convert records to samples with error handling
    import warnings

    samples = []
    skipped_count = 0

    for record in raw_dataset:
        try:
            sample = record_to_mcq_sample(record)
            if sample is not None:
                samples.append(sample)
        except ValueError as e:
            # Validation error - skip this record with warning
            skipped_count += 1
            record_id = record.get("idx", record.get("id", f"record_{skipped_count}"))
            warnings.warn(
                f"Skipping invalid MCQ record (id={record_id}) in {name}: {e}",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            # Unexpected error - skip but warn about it
            skipped_count += 1
            record_id = record.get("idx", record.get("id", f"record_{skipped_count}"))
            warnings.warn(
                f"Unexpected error processing MCQ record (id={record_id}) in {name}: {e}",
                UserWarning,
                stacklevel=2,
            )

    if skipped_count > 0:
        warnings.warn(
            f"Skipped {skipped_count} invalid record(s) in {name}. "
            f"Proceeding with {len(samples)} valid samples.",
            UserWarning,
            stacklevel=2,
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
        dataset=samples,
        solver=solver,
        scorer=scorer,
        config=config if config else GenerateConfig(),
        epochs=epochs,
    )
