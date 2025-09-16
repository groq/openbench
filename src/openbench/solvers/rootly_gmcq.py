"""Rootly GMCQ solver."""

from __future__ import annotations

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model


@solver
def rootly_gmcq_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = await model.generate(input=state.input)
        state.messages.append(resp.choices[0].message)
        return state

    return solve
