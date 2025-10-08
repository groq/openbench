"""
Breakpoint - Code repair benchmark for evaluating LLM reasoning in debugging

Breakpoint tests LLMs' ability to diagnose, explore, and repair code in Python/Pytest
repositories. It provides two evaluation modes:

- **Remove mode**: Function bodies are deleted; the model must reconstruct them
- **Discovery mode**: Subtle corruptions are introduced; the model must locate and fix them

The benchmark focuses on system-level reasoning by requiring models to:
1. Understand code context within a repository
2. Interpret test failure messages
3. Locate and fix bugs iteratively

Dataset: https://huggingface.co/datasets/uzpg/breakpoint
Paper: http://arxiv.org/pdf/2506.00172

Sample usage:
```bash
# Run remove mode (498 problems)
bench eval breakpoint_remove --model "groq/llama-3.1-70b" --limit 10

# Run discovery mode (269 problems)
bench eval breakpoint_discovery --model "groq/llama-3.1-70b" --limit 10
```

Citation:
@article{hariharan2025breakpoint,
    title={Breakpoint: Scalable evaluation of system-level reasoning in LLM code agents},
    author={Hariharan, Kaivalya and Girit, Uzay and Wang, Atticus and Andreas, Jacob},
    journal={arXiv preprint arXiv:2506.00172},
    year={2025}
}
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import system_message
from typing import Any

from openbench.solvers.breakpoint_solver import breakpoint_solver
from openbench.scorers.breakpoint_scorer import breakpoint_scorer


# HuggingFace dataset URLs
BREAKPOINT_REMOVE_URL = (
    "https://huggingface.co/datasets/uzpg/breakpoint/resolve/main/data/remove-data.json"
)
BREAKPOINT_DISCOVERY_URL = "https://huggingface.co/datasets/uzpg/breakpoint/resolve/main/data/discovery-data.json"


def record_to_sample_remove(record: dict[str, Any]) -> Sample:
    """Convert a remove-mode record to an Inspect AI Sample.

    In remove mode, the function body has been deleted and the model must
    reconstruct it based on:
    - The repository context
    - Test failure information
    - Function signature and location

    Args:
        record: Dictionary containing repo info, file path, function name,
                test info, and complexity metrics

    Returns:
        Sample with the problem statement and metadata
    """
    repo = record["repo"]
    fpath = record["fpath"]
    function_name = record["function_name"]
    test_info = record.get("test_info", {})

    # Create problem description
    prompt = f"""You are debugging code in a Python repository.

**Repository**: {repo.get("name", "unknown")}
**File**: {fpath}
**Function**: `{function_name}`

The function body has been removed and needs to be reconstructed.

**Test Command**: {repo.get("test_command", "pytest")}
**Test Status**: {"FAILING" if not test_info.get("success", True) else "PASSING"}
**Failed Tests**: {test_info.get("failed", 0)}
**Passing Tests**: {test_info.get("passed", 0)}

"""

    # Add failure information if available
    failures = test_info.get("failures_info", [])
    if failures:
        prompt += "\n**Test Failures**:\n"
        for i, failure in enumerate(failures[:3], 1):  # Limit to first 3 failures
            prompt += f"\n{i}. {failure[:500]}\n"  # Limit each failure to 500 chars

    prompt += """
Your task is to reconstruct the missing function body. Provide only the complete function implementation.
"""

    return Sample(
        input=prompt,
        target="",  # No ground truth available
        id=f"{repo.get('name', 'unknown')}_{function_name}",
        metadata={
            "repo_name": repo.get("name"),
            "repo_url": repo.get("url"),
            "repo_commit": repo.get("commit"),
            "fpath": fpath,
            "function_name": function_name,
            "test_command": repo.get("test_command", "pytest"),
            "baseline_failures": test_info.get("failed", 1),
            "mode": "remove",
            "corruption_code": None,
        },
    )


def record_to_sample_discovery(record: dict[str, Any]) -> Sample:
    """Convert a discovery-mode record to an Inspect AI Sample.

    In discovery mode, a subtle corruption has been introduced and the model
    must locate and fix it.

    Args:
        record: Dictionary containing repo info, file path, function name,
                test info, and corruption details

    Returns:
        Sample with the problem statement and metadata
    """
    repo = record["repo"]
    fpath = record["fpath"]
    function_name = record["function_name"]
    test_info = record.get("test_info", {})
    corruption = record.get("corruption", {})

    # Create problem description
    prompt = f"""You are debugging code in a Python repository.

**Repository**: {repo.get("name", "unknown")}
**File**: {fpath}
**Function**: `{function_name}`

A subtle bug has been introduced in this function and needs to be fixed.

**Corrupted Code**:
```python
{corruption.get("code", "")}
```

**Test Command**: {repo.get("test_command", "pytest")}
**Test Status**: {"FAILING" if not test_info.get("success", True) else "PASSING"}
**Failed Tests**: {test_info.get("failed", 0)}
**Passing Tests**: {test_info.get("passed", 0)}

"""

    # Add failure information if available
    failures = test_info.get("failures_info", [])
    if failures:
        prompt += "\n**Test Failures**:\n"
        for i, failure in enumerate(failures[:3], 1):  # Limit to first 3 failures
            prompt += f"\n{i}. {failure[:500]}\n"  # Limit each failure to 500 chars

    prompt += """
Your task is to identify and fix the bug. Provide the corrected function implementation.
"""

    return Sample(
        input=prompt,
        target="",  # No ground truth available
        id=f"{repo.get('name', 'unknown')}_{function_name}",
        metadata={
            "repo_name": repo.get("name"),
            "repo_url": repo.get("url"),
            "repo_commit": repo.get("commit"),
            "fpath": fpath,
            "function_name": function_name,
            "test_command": repo.get("test_command", "pytest"),
            "baseline_failures": test_info.get("failed", 1),
            "corruption_score": corruption.get("score", 0),
            "corruption_code": corruption.get("code"),
            "mode": "discovery",
        },
    )


@task
def breakpoint_remove() -> Task:
    """
    Breakpoint Remove Mode - Function reconstruction task

    Tests the model's ability to reconstruct deleted function bodies based on:
    - Repository context
    - Test failure messages
    - Function signatures and locations

    Dataset: 498 problems from real Python repositories
    """
    return Task(
        dataset=json_dataset(
            json_file=BREAKPOINT_REMOVE_URL,
            sample_fields=record_to_sample_remove,
            auto_id=True,
        ),
        solver=[
            system_message(
                "You are an expert Python developer helping to debug and fix code. "
                "Analyze the problem carefully, consider the test failures, and provide "
                "a complete, working implementation."
            ),
            breakpoint_solver(),
        ],
        scorer=breakpoint_scorer(),
        sandbox="local",
    )


@task
def breakpoint_discovery() -> Task:
    """
    Breakpoint Discovery Mode - Bug location and repair task

    Tests the model's ability to locate and fix subtle bugs in code based on:
    - Corrupted function code
    - Test failure messages
    - Repository context

    Dataset: 269 problems from real Python repositories
    """
    return Task(
        dataset=json_dataset(
            json_file=BREAKPOINT_DISCOVERY_URL,
            sample_fields=record_to_sample_discovery,
            auto_id=True,
        ),
        solver=[
            system_message(
                "You are an expert Python developer helping to debug and fix code. "
                "Analyze the corrupted code carefully, identify the bug, and provide "
                "a corrected implementation."
            ),
            breakpoint_solver(),
        ],
        scorer=breakpoint_scorer(),
        sandbox="local",
    )
