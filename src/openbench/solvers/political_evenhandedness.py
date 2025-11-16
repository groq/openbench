"""Solver that generates paired responses for political even-handedness prompts."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver


@solver
def paired_prompt_solver(prompt_order: str = "ab"):
    """
    Generate responses for both prompts in a political prompt pair.

    Args:
        prompt_order: Order to present prompts to the model. "ab" (default) first
        generates prompt_a then prompt_b; "ba" reverses the order.
    """

    normalized_order = prompt_order.lower()
    if normalized_order not in {"ab", "ba"}:
        raise ValueError("prompt_order must be 'ab' or 'ba'")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}

        metadata: Dict[str, str] = state.metadata
        prompt_a = str(metadata.get("prompt_a", "")).strip()
        prompt_b = str(metadata.get("prompt_b", "")).strip()

        prompts: List[Tuple[str, str]] = [("A", prompt_a), ("B", prompt_b)]
        if normalized_order == "ba":
            prompts.reverse()

        responses: Dict[str, str] = {}
        generations: Dict[str, Dict[str, Any]] = {}

        for label, prompt_text in prompts:
            user_message = ChatMessageUser(content=prompt_text)

            state_copy = copy.deepcopy(state)
            state_copy.messages = [user_message]

            result_state = await generate(state_copy)
            completion = result_state.output.completion
            responses[label] = completion

            generations[label] = {
                "prompt": prompt_text,
                "response": completion,
            }

        # Restore ordering when recording metadata
        state.metadata["model_response_a"] = responses.get("A", "")
        state.metadata["model_response_b"] = responses.get("B", "")
        state.metadata["prompt_order"] = normalized_order

        state.output.completion = (
            f"Prompt A Response:\n{responses.get('A', '').strip()}\n\n"
            f"Prompt B Response:\n{responses.get('B', '').strip()}"
        ).strip()

        return state

    return solve


__all__ = ["paired_prompt_solver"]
