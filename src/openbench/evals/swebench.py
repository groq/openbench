from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message
from openbench.scorers.swebench import swebench_format_scorer, swebench_docker_scorer
from openbench.datasets.swebench import (
    get_swebench_dataset,
    get_swebench_lite_dataset,
    get_swebench_verified_dataset,
)
from typing import Optional

# System message for SWE-bench tasks
SWEBENCH_SYSTEM_MESSAGE = """
You are an expert software engineer with extensive experience in debugging, 
code analysis, and patch generation. You excel at understanding complex 
codebases and resolving GitHub issues efficiently.

When presented with a software issue, you should:
1. Carefully analyze the problem statement
2. Consider the repository context and structure
3. Generate a precise patch that resolves the issue
4. Ensure your patch follows proper diff/patch format conventions

Your response should contain a well-formatted patch that can be applied 
to resolve the described issue.
""".strip()


@task
def swebench(
    instruction_prompt: Optional[str] = None, use_docker: bool = False
) -> Task:
    """
    SWE-bench: Full benchmark with 2,294 real-world GitHub issues.

    Evaluates language models on their ability to resolve real GitHub issues
    by generating patches. This is the complete dataset covering 12 Python
    repositories with diverse software engineering challenges.

    Based on: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
    Original repository: https://github.com/SWE-bench/SWE-bench

    Args:
        instruction_prompt (str, optional): Custom instruction prompt to prepend
            to each problem statement. If None, uses default SWE-bench instruction.
        use_docker (bool): Whether to use Docker-based validation for full accuracy
            testing. Default False uses format validation only for faster evaluation.

    Returns:
        Task: The configured SWE-bench task.
    """
    dataset = (
        get_swebench_dataset()
        if instruction_prompt is None
        else get_swebench_dataset(instruction_prompt=instruction_prompt)
    )

    scorer = swebench_docker_scorer() if use_docker else swebench_format_scorer()

    return Task(
        dataset=dataset,
        solver=[system_message(SWEBENCH_SYSTEM_MESSAGE), generate()],
        scorer=scorer,
        config=GenerateConfig(
            temperature=0.3,  # Lower temperature for more consistent patches
            max_tokens=4000,  # Allow for longer patches
        ),
        epochs=Epochs(1),  # Single epoch due to resource intensity
    )


@task
def swebench_lite(
    instruction_prompt: Optional[str] = None, use_docker: bool = False
) -> Task:
    """
    SWE-bench Lite: Streamlined version with 300 curated tasks.

    A more manageable subset of SWE-bench designed for efficient evaluation
    while maintaining representative coverage of software engineering challenges.
    Contains 300 test cases from 11 popular Python projects.

    Based on: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
    Original repository: https://github.com/SWE-bench/SWE-bench

    Args:
        instruction_prompt (str, optional): Custom instruction prompt to prepend
            to each problem statement. If None, uses default SWE-bench instruction.
        use_docker (bool): Whether to use Docker-based validation for full accuracy
            testing. Default False uses format validation only for faster evaluation.

    Returns:
        Task: The configured SWE-bench Lite task.
    """
    dataset = (
        get_swebench_lite_dataset()
        if instruction_prompt is None
        else get_swebench_lite_dataset(instruction_prompt=instruction_prompt)
    )

    scorer = swebench_docker_scorer() if use_docker else swebench_format_scorer()

    return Task(
        dataset=dataset,
        solver=[system_message(SWEBENCH_SYSTEM_MESSAGE), generate()],
        scorer=scorer,
        config=GenerateConfig(
            temperature=0.3,  # Lower temperature for more consistent patches
            max_tokens=4000,  # Allow for longer patches
        ),
        epochs=Epochs(1),  # Single epoch due to resource intensity
    )


@task
def swebench_verified(
    instruction_prompt: Optional[str] = None, use_docker: bool = False
) -> Task:
    """
    SWE-bench Verified: Human-validated subset with 500 high-quality tasks.

    A carefully curated and human-validated subset of SWE-bench, released by OpenAI.
    Each task has been reviewed and validated by human experts, ensuring high
    quality and reliable evaluation metrics.

    Based on: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
    Original repository: https://github.com/SWE-bench/SWE-bench
    Verified variant: https://openai.com/index/introducing-swe-bench-verified/

    Args:
        instruction_prompt (str, optional): Custom instruction prompt to prepend
            to each problem statement. If None, uses default SWE-bench instruction.
        use_docker (bool): Whether to use Docker-based validation for full accuracy
            testing. Default False uses format validation only for faster evaluation.

    Returns:
        Task: The configured SWE-bench Verified task.
    """
    dataset = (
        get_swebench_verified_dataset()
        if instruction_prompt is None
        else get_swebench_verified_dataset(instruction_prompt=instruction_prompt)
    )

    scorer = swebench_docker_scorer() if use_docker else swebench_format_scorer()

    return Task(
        dataset=dataset,
        solver=[system_message(SWEBENCH_SYSTEM_MESSAGE), generate()],
        scorer=scorer,
        config=GenerateConfig(
            temperature=0.3,  # Lower temperature for more consistent patches
            max_tokens=4000,  # Allow for longer patches
        ),
        epochs=Epochs(1),  # Single epoch due to resource intensity
    )
