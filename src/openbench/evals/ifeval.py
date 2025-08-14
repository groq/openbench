"""
Instruction Following Evaluation (IFEval) for Large Language Models

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma,
Sujoy Basu, Yi Luan, Denny Zhou, Le Hou
https://arxiv.org/abs/2311.07911

Based on: https://github.com/google-research/google-research/tree/master/instruction_following_eval
with code adapted from this repository.

Evaluates a model's ability to follow specific instructions and constraints
in text generation tasks.

# eval all constraint types with a strict scorer type:
bench eval ifeval --model "groq/llama-3.1-8b-instant" -T scorer_type=strict
"""

from colorama import Fore, Style
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import solver, Solver
from inspect_ai.model import get_model
from inspect_ai.model import ChatMessageUser

from openbench.datasets.ifeval import get_ifeval_dataset
from openbench.scorers.ifeval.scorer import custom_scorer


@solver
def custom_solver() -> Solver:
    """Custom solver for IFEval."""

    model = get_model()

    async def solve(state, generate):
        resp = await model.generate(
            input=[
                ChatMessageUser(content=state.input),
            ],
        )

        state.messages.append(resp.choices[0].message)
        return state

    return solve


@task
def ifeval(scorer_type: str | None = None) -> Task:
    """Inspect Task implementation for IFEval"""
    if scorer_type is None:
        print(
            f"{Fore.YELLOW}⚠️  No scorer type provided for IFEval through OpenBench. Defaulting to 'strict'.{Style.RESET_ALL}"
        )
        scorer_type = "strict"
    return Task(
        dataset=get_ifeval_dataset(),
        solver=custom_solver(),
        scorer=custom_scorer(scorer_type),
        config=GenerateConfig(),
    )
