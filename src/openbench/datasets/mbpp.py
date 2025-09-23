from inspect_ai.dataset import Sample, Dataset, hf_dataset
from typing import Any, Callable

INSTRUCTION_PROMPT = """
You are an expert Python programmer, and here is your task: {prompt} 
Your code should pass these tests:

{tests}

Return your code in the following format, include [BEGIN] and [DONE] tags:
[BEGIN]
<ONLY include your code here, no commentary, no imports, no tests>
[DONE]
""".strip()


def record_to_sample(
    instruction_prompt: str = INSTRUCTION_PROMPT,
) -> Callable[[dict[str, Any]], Sample]:
    """
    Convert a MBPP record to a Sample for evaluation.

    Args:
        instruction_prompt (str): The prompt to wrap the code problem in.

    Returns:
        Callable[[dict[str, Any]], Sample]: Function to convert a record dict to a Sample.
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample:
        # newline separated in preparation for sandbox execution
        all_tests = "\n" + "\n".join(
            record["test_list"] + record.get("challenge_test_list", [])
        )
        return Sample(
            id=record["task_id"],
            input=instruction_prompt.format(
                # key text for full, prompt for sanitized
                prompt=record.get("text") or record.get("prompt"),
                tests=all_tests,
            ),
            target=record["code"],
            metadata={
                # key test_setup_code for full, test_imports for sanitized
                "test_setup_code": "\n".join(
                    record.get("test_setup_code") or record.get("test_imports") or ""
                ),
                "tests": all_tests,
                "source_file": record.get("source_file", "Unknown"),
                "task_id": record["task_id"],
            },
        )

    return _record_to_sample


def get_mbpp_dataset(subset: str = "full", split: str = "test") -> Dataset:
    """
    Load the MBPP dataset.

    Args:
        subset: Which subset to load ("full" (default), "sanitized")
        split: Which split to load ("test" (default), "train", "validation", "prompt")
            (See Google Research Dataset Documentation for more information on splits)
    Returns:
        Dataset configured for MBPP evaluation
    """
    if subset not in ("full", "sanitized"):
        raise ValueError(
            f"Invalid subset '{subset}'. Available subsets are 'full' or 'sanitized'."
        )

    return hf_dataset(
        path="google-research-datasets/mbpp",
        name=subset,
        split=split,
        sample_fields=record_to_sample(),
    )
