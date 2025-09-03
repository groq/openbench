from inspect_ai.solver import Solver, solver
from inspect_ai.model import GenerateConfig


@solver
def generate_with_dynamic_tokens() -> Solver:
    """Custom solver that sets max_tokens per sample based on metadata."""

    async def solve(state, generate_fn):
        sample_max_tokens = state.metadata.get("max_output_tokens", 8192)

        config = GenerateConfig(temperature=0.0, max_tokens=sample_max_tokens)

        return await generate_fn(state, config=config)

    return solve
