"""
Breakpoint solver for parsing function implementations from model completions.
"""

from inspect_ai.solver import Solver, TaskState, solver
from openbench.utils.breakpoint_utils import parse_function_from_completion


@solver
def breakpoint_solver() -> Solver:
    """
    Parse function code from model completion.

    Extracts the function implementation from the model's response
    and stores it in state metadata for the scorer to use.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        # Generate completion
        state = await generate(state)

        # Extract function code from completion
        completion = state.output.completion
        function_code = parse_function_from_completion(completion)

        # Store in metadata for scorer
        if not state.metadata:
            state.metadata = {}

        state.metadata["parsed_function_code"] = function_code

        return state

    return solve
