from typing import Any, Callable

from inspect_ai.dataset import Sample, Dataset, hf_dataset

HUMANEVAL_INSTRUCTION = """
    Read the following function signature and docstring, and fully implement
    the function described. Your response should only contain the code for
    this function.
    """.strip()


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
def record_to_sample(
    instruction_prompt: str = HUMANEVAL_INSTRUCTION,
) -> Callable[[dict[str, Any]], Sample]:
    """
    Convert a HumanEval record to a Sample for evaluation.

    Args:
        instruction_prompt (str): The prompt to prepend to the code problem.

    Returns:
        Callable[[dict[str, Any]], Sample]: Function to convert a record dict to a Sample.
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            id=record["task_id"],
            input=instruction_prompt + record["prompt"],
            target=record["canonical_solution"],
            metadata={
                "prompt": record["prompt"],
                "test": record["test"],
                "entry_point": record["entry_point"],
            },
        )

    return _record_to_sample


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
def get_humaneval_dataset(instruction_prompt: str = HUMANEVAL_INSTRUCTION) -> Dataset:
    """
    Load the HumanEval dataset for evaluation.

    Args:
        instruction_prompt (str): The prompt to prepend to the code problem.

    Returns:
        Dataset: The HumanEval dataset.
    """
    return hf_dataset(
        path="openai_humaneval",
        split="test",
        sample_fields=record_to_sample(instruction_prompt=instruction_prompt),
    )
