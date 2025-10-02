from typing import Callable, Optional
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageSystem,
    ContentImage,
    ContentText,
    Model,
    get_model,
)
import json


@solver
def mmstar_solver(base_model: Optional[str] = None) -> Callable:
    """Solver that queries candidate model with and without vision inputs."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        candidate: Model = get_model()
        base: Optional[Model] = get_model(base_model) if base_model else None

        question = state.metadata["question"].strip()
        image_uri = state.metadata["image_uri"].strip()

        # system message (from original implementation)
        system_msg = ChatMessageSystem(
            content="Return your answer as a single uppercase letter with no additional text. Even if you are unsure, return your best guess as a single uppercase letter."
        )

        # -- vision candidate with vision input --
        with_vision_message = ChatMessageUser(
            content=[ContentImage(image=image_uri), ContentText(text=question)]
        )
        with_vision_response = await candidate.generate(
            [system_msg, with_vision_message]
        )
        with_vision_answer = with_vision_response.completion

        # -- vision candidate without vision input --
        without_vision_message = ChatMessageUser(content=[ContentText(text=question)])
        without_vision_response = await candidate.generate(
            [system_msg, without_vision_message]
        )
        without_vision_answer = without_vision_response.completion

        # -- base text model output (optional) --

        if base:
            base_message = ChatMessageUser(content=[ContentText(text=question)])
            try:
                base_response = await base.generate([system_msg, base_message])
            except Exception as e:
                raise ValueError(f"Base model response not generated: {e}")
            base_answer = base_response.completion
        else:
            base_answer = ""

        # add evaluation metadata for scoring
        state.output.completion = json.dumps(
            {
                "with_vision_answer": with_vision_answer,
                "without_vision_answer": without_vision_answer,
                "text_base_answer": base_answer,
                "category": state.metadata["category"],
                "subcategory": state.metadata["subcategory"],
                "image_path": state.metadata["image_path"],
                "question": state.metadata["question"],
                "base_model": base_model,
            },
            indent=2,
        )
        return state

    return solve
