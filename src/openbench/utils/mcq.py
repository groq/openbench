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
from inspect_ai.dataset import Sample, csv_dataset, hf_dataset, json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai import Task, Epochs
from inspect_ai.scorer import Scorer
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


# ----------- GLOBAL MCQ SCORING CONFIGURATION -----------
# NOTE: These settings apply to MCQEval infrastructure only.
# For custom scoring outside MCQEval, use create_mcq_scorer() directly.

USE_MODEL_GRADING = False
"""
Global flag to enable model-graded MCQ scoring (default: False).

When False (default), uses regex-based answer extraction (fast, but may have accuracy
issues with verbose model reasoning).

When True, uses an LLM to grade MCQ answers, which is more reliable for models that
provide extensive reasoning before their final answer. Enable with --model-graded flag.

This setting applies to all tasks created via MCQEval.
"""

MCQ_GRADING_MODEL = "groq/openai/gpt-oss-20b"
"""
Default model to use for MCQ answer grading (default: "groq/openai/gpt-oss-20b").

This model is used to determine the model's final answer when USE_MODEL_GRADING=True.
Set to None to use the same model being evaluated.
"""


# ----------- MCQ DETECTION -----------


def is_mcq_task(task: Task) -> bool:
    """Detect if a task uses MCQ scoring by inspecting its scorer.

    This function checks the actual implementation to determine if a task
    uses MCQEval infrastructure or MCQ-based scorers. It's used both for:
    1. Runtime validation (--model-graded flag)
    2. Build-time tagging (auto-generate benchmarks script)

    Args:
        task: The loaded Inspect AI task object

    Returns:
        True if the task uses MCQ scoring (either regex or model-graded)

    Examples:
        >>> from openbench.config import load_task
        >>> task_fn = load_task("mmlu")
        >>> task = task_fn()
        >>> is_mcq_task(task)
        True
        >>> task_fn = load_task("humaneval")
        >>> task = task_fn()
        >>> is_mcq_task(task)
        False
    """
    # Get the scorer from the task
    scorer_obj = task.scorer if hasattr(task, "scorer") else None
    if scorer_obj is None:
        return False

    # Handle case where scorer is a list (common pattern in Inspect AI)
    # Extract the first scorer from the list
    scorer = (
        scorer_obj[0] if isinstance(scorer_obj, list) and scorer_obj else scorer_obj
    )
    if scorer is None or (isinstance(scorer_obj, list) and not scorer_obj):
        return False

    # Check if it's a Scorer instance
    if isinstance(scorer, Scorer):
        # Check the scorer's name - MCQ scorers have specific naming patterns
        scorer_name = scorer.name if hasattr(scorer, "name") else ""

        # Scorers from create_mcq_scorer are named "mcq_scorer"
        # model_graded_fact scorers are named "model_graded_fact"
        if scorer_name in ["mcq_scorer", "model_graded_fact"]:
            return True

        # Also check if the scorer function name contains mcq-related terms
        # This handles pre-configured scorers like mmlu_simple_eval_scorer, tumlu_simple_eval_scorer
        if hasattr(scorer, "_scorer") and hasattr(scorer._scorer, "__name__"):
            func_name = scorer._scorer.__name__.lower()
            if "mcq" in func_name:
                return True

    # Check if it's a callable (scorer function directly)
    if callable(scorer):
        # Check function name for mcq-related terms
        func_name = getattr(scorer, "__name__", "").lower()
        if "mcq" in func_name or "score" in func_name:
            # Additional check: look for create_mcq_scorer in the qualified name
            qual_name = getattr(scorer, "__qualname__", "")
            if "mcq_scorer" in qual_name or "create_mcq_scorer" in qual_name:
                return True

    return False


def get_mcq_benchmarks(include_alpha: bool = False) -> List[str]:
    """Get a list of all MCQ benchmarks by static analysis of eval files.

    This function uses grep to search for MCQ-related patterns in eval files,
    which is much faster than loading all benchmarks.

    Detects MCQ benchmarks by searching for:
    - MCQEval( - tasks using the MCQEval factory
    - create_mcq_scorer - tasks using MCQ scorers directly

    Used by:
    1. CLI validation for --model-graded flag
    2. Auto-generate script for adding "mcq" tags

    Args:
        include_alpha: Whether to include alpha/experimental benchmarks

    Returns:
        List of benchmark names that use MCQ scoring

    Note:
        This uses static analysis (grep) which is very fast (~1 second)
        instead of loading all benchmarks (~3 minutes).
    """
    import subprocess
    from pathlib import Path

    # Avoid circular import
    from openbench.config import get_all_benchmarks

    # Get the evals directory
    evals_dir = Path(__file__).parent.parent / "evals"

    # Search for MCQ patterns in eval files
    mcq_files = set()

    try:
        # Search for MCQEval usage
        result = subprocess.run(
            ["grep", "-l", "-r", "MCQEval(", str(evals_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            mcq_files.update(result.stdout.strip().split("\n"))
    except Exception:
        pass

    try:
        # Search for create_mcq_scorer usage
        result = subprocess.run(
            ["grep", "-l", "-r", "create_mcq_scorer", str(evals_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            mcq_files.update(result.stdout.strip().split("\n"))
    except Exception:
        pass

    # Get all benchmarks and their file paths
    all_benchmarks = get_all_benchmarks(include_alpha=include_alpha)

    # Map eval files to benchmark names
    mcq_benchmarks = []
    for benchmark_name, metadata in all_benchmarks.items():
        # Extract eval filename from module_path (e.g., "openbench.evals.mmlu" -> "mmlu.py")
        if hasattr(metadata, "module_path"):
            # Split module path and get the filename part
            parts = metadata.module_path.split(".")
            if len(parts) >= 3 and parts[0] == "openbench" and parts[1] == "evals":
                eval_filename = parts[2] + ".py"
                # Check if this file uses MCQ patterns
                if any(
                    str(f).endswith(f"/{eval_filename}")
                    or str(f).endswith(f"\\{eval_filename}")
                    for f in mcq_files
                ):
                    mcq_benchmarks.append(benchmark_name)

    return mcq_benchmarks


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

    Scoring behavior is controlled by global configuration:
    - `USE_MODEL_GRADING`: Enable model-graded scoring (default: True)
    - `MCQ_GRADING_MODEL`: Model to use for grading (default: "groq/openai/gpt-oss-20b")

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
    if dataset_type == "hf":
        if split is None:
            raise ValueError("For dataset_type='hf', you must provide split")
        dataset = hf_dataset(
            dataset_path,
            split=split,
            sample_fields=record_to_mcq_sample,
            auto_id=auto_id,
            name=subset_name,  # subset name
            **(dataset_kwargs or {}),
        )
    elif dataset_type == "csv":
        dataset = csv_dataset(
            csv_file=dataset_path,
            sample_fields=record_to_mcq_sample,
            auto_id=auto_id,
            **(dataset_kwargs or {}),
        )
    elif dataset_type == "json":
        dataset = json_dataset(
            json_file=dataset_path,
            sample_fields=record_to_mcq_sample,
            auto_id=auto_id,
            **(dataset_kwargs or {}),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    solver = [generate()]
    if prompt_template:
        solver = [system_message(prompt_template), generate()]

    # Use global MCQ scoring configuration
    scorer = create_mcq_scorer(
        group_keys=group_keys,
        additional_metrics=additional_metrics,
        use_model_grading=USE_MODEL_GRADING,
        grading_model=MCQ_GRADING_MODEL,
    )()

    return Task(
        name=name,
        dataset=dataset,
        solver=solver,
        scorer=scorer,
        config=config if config else GenerateConfig(),
        epochs=epochs,
    )
