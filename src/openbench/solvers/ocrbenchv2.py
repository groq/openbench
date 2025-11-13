"""OCRBench v2 solver for visual text localization and reasoning tasks."""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.solver import Generate, TaskState, solver


@solver
def ocrbenchv2_solver():
    """
    Solver that sends image + question to the model with a system prompt.

    The model receives:
    - System prompt: Instructions to provide concise answers
    - Image (as base64 data URI)
    - Question text

    The system prompt guides the model to provide direct answers without
    additional explanation, which improves scoring accuracy.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get image URI from metadata
        image_uri = state.metadata["image_uri"]

        # Add system prompt to guide concise responses and preserve formatting
        state.messages = [
            ChatMessageUser(
                content=[
                    ContentText(
                        text="Provide the final answer with no additional text or explanation. Preserve exact formatting including capitalization, spacing, and line breaks as shown in the image. For multiple choice questions, provide ONLY the single letter option (e.g., 'A' not 'A, B' or 'A and B')."
                    ),
                    ContentImage(image=image_uri),
                    ContentText(text=state.input_text),
                ]
            )
        ]

        # Generate response
        result = await generate(state)
        state.output.completion = result.output.completion

        return state

    return solve
