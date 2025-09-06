from typing import Any, Callable

from inspect_ai.dataset import Sample, Dataset, hf_dataset

# System prompt for SWE-bench tasks
SWEBENCH_INSTRUCTION = """
You are a software engineer tasked with resolving a GitHub issue. 
You will be given a problem statement describing an issue in a codebase.
Your task is to generate a patch (code changes) that resolves the issue.

Please provide your solution as a unified diff patch format.
""".strip()


def record_to_sample(
    instruction_prompt: str = SWEBENCH_INSTRUCTION,
) -> Callable[[dict[str, Any]], Sample]:
    """
    Convert a SWE-bench record to a Sample for evaluation.

    Args:
        instruction_prompt (str): The prompt to prepend to the problem statement.

    Returns:
        Callable[[dict[str, Any]], Sample]: Function to convert a record dict to a Sample.
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample:
        # Construct the input with problem statement and repo context
        problem_input = f"{instruction_prompt}\n\n"
        problem_input += f"Repository: {record.get('repo', 'Unknown')}\n"
        problem_input += f"Issue: {record.get('problem_statement', '')}\n\n"

        # Add hints if available
        if record.get("hints_text"):
            problem_input += f"Hints: {record.get('hints_text')}\n\n"

        problem_input += "Please provide a patch to resolve this issue:"

        return Sample(
            id=record.get("instance_id", "unknown"),
            input=problem_input,
            target=record.get("patch", ""),
            metadata={
                "repo": record.get("repo"),
                "base_commit": record.get("base_commit"),
                "test_patch": record.get("test_patch"),
                "problem_statement": record.get("problem_statement"),
                "hints_text": record.get("hints_text", ""),
                "created_at": record.get("created_at"),
                "version": record.get("version", ""),
                "environment_setup_commit": record.get("environment_setup_commit"),
                "FAIL_TO_PASS": record.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": record.get("PASS_TO_PASS", ""),
            },
        )

    return _record_to_sample


def get_swebench_dataset(instruction_prompt: str = SWEBENCH_INSTRUCTION) -> Dataset:
    """
    Load the SWE-bench dataset (full version) for evaluation.

    This dataset contains 2,294 Issue-Pull Request pairs from 12 Python repositories.

    Args:
        instruction_prompt (str): The prompt to prepend to the problem statement.

    Returns:
        Dataset: The SWE-bench dataset.
    """
    return hf_dataset(
        path="princeton-nlp/SWE-bench",
        split="test",
        sample_fields=record_to_sample(instruction_prompt=instruction_prompt),
    )


def get_swebench_lite_dataset(
    instruction_prompt: str = SWEBENCH_INSTRUCTION,
) -> Dataset:
    """
    Load the SWE-bench Lite dataset for evaluation.

    This dataset contains 300 test Issue-Pull Request pairs from 11 popular Python projects.
    It's a streamlined version designed for more manageable experimentation.

    Args:
        instruction_prompt (str): The prompt to prepend to the problem statement.

    Returns:
        Dataset: The SWE-bench Lite dataset.
    """
    return hf_dataset(
        path="princeton-nlp/SWE-bench_Lite",
        split="test",
        sample_fields=record_to_sample(instruction_prompt=instruction_prompt),
    )


def get_swebench_verified_dataset(
    instruction_prompt: str = SWEBENCH_INSTRUCTION,
) -> Dataset:
    """
    Load the SWE-bench Verified dataset for evaluation.

    This dataset contains 500 human-validated test Issue-Pull Request pairs from
    popular Python repositories. Each task has been carefully reviewed and validated
    by human experts.

    Args:
        instruction_prompt (str): The prompt to prepend to the problem statement.

    Returns:
        Dataset: The SWE-bench Verified dataset.
    """
    return hf_dataset(
        path="princeton-nlp/SWE-bench_Verified",
        split="test",
        sample_fields=record_to_sample(instruction_prompt=instruction_prompt),
    )
