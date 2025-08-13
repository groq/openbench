"""
CRUXEval: Code Reasoning, Understanding, and eXecution Evaluation

A benchmark of Python functions and input-output pairs. The benchmark consists of two tasks:
1. CRUXEval-I (Input Prediction): Given a Python function and its output, predict the input
2. CRUXEval-O (Output Prediction): Given a Python function and its input, predict the output

Based on: https://github.com/facebookresearch/cruxeval
"""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.cruxeval import get_dataset
from openbench.scorers.cruxeval import cruxeval_scorer


# Number of epochs to run each problem
NUM_EPOCHS = 2


@task
def cruxeval_input(use_cot: bool = False) -> Task:
    """
    CRUXEval-I: Input Prediction task.
    Given a Python function and its output, predict the input.

    Args:
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Task configured for CRUXEval-I
    """
    dataset = get_dataset(task_type="input", use_cot=use_cot)

    return Task(
        dataset=dataset,
        epochs=NUM_EPOCHS,
        solver=[generate()],
        scorer=cruxeval_scorer(),
        name=f"cruxeval_input{'_cot' if use_cot else ''}",
        config=GenerateConfig(
            max_tokens=1000,
            temperature=0.0,
            top_p=1.0,
        ),
    )


@task
def cruxeval_output(use_cot: bool = False) -> Task:
    """
    CRUXEval-O: Output Prediction task.
    Given a Python function and its input, predict the output.

    Args:
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Task configured for CRUXEval-O
    """
    dataset = get_dataset(task_type="output", use_cot=use_cot)

    return Task(
        dataset=dataset,
        epochs=NUM_EPOCHS,
        solver=[generate()],
        scorer=cruxeval_scorer(),
        name=f"cruxeval_output{'_cot' if use_cot else ''}",
        config=GenerateConfig(
            max_tokens=1000,
            temperature=0.0,
            top_p=1.0,
        ),
    )


@task
def cruxeval_input_cot() -> Task:
    """
    CRUXEval-I with Chain-of-Thought: Input Prediction task.
    Given a Python function and its output, predict the input using step-by-step reasoning.

    Returns:
        Task configured for CRUXEval-I with CoT
    """
    return cruxeval_input(use_cot=True)


@task
def cruxeval_output_cot() -> Task:
    """
    CRUXEval-O with Chain-of-Thought: Output Prediction task.
    Given a Python function and its input, predict the output using step-by-step reasoning.

    Returns:
        Task configured for CRUXEval-O with CoT
    """
    return cruxeval_output(use_cot=True)
