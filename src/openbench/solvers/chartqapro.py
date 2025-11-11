"""ChartQAPro multi-turn solver with category-specific prompts."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
)
from inspect_ai.solver import Generate, TaskState, solver

from openbench.prompts.chartqapro import (
    PromptStrategy,
    format_question_with_paragraph,
    get_system_prompt,
)
from openbench.scorers.chartqapro import extract_final_answer


@solver
def chartqapro_solver(prompt_strategy: PromptStrategy = "direct"):
    """
    Multi-turn solver for ChartQAPro with category-specific prompts.

    Handles 1-7 sequential questions with shared conversation context.
    Image is sent only in the first turn (following best practices).

    For Conversational questions:
    - All questions are asked sequentially
    - Each answer builds on previous context
    - Only the LAST answer is scored

    For other question types:
    - Typically single question
    - Can handle multiple questions if present

    Args:
        prompt_strategy: Prompting approach
            - "direct": Concise answer only (default)
            - "cot": Chain-of-thought reasoning
            - "pot": Program-of-thought (executable Python)

    Returns:
        Solver function that processes samples
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            # Extract metadata
            questions = state.metadata["questions"]
            img_ref = state.metadata["image_data_uri"]
            question_type = state.metadata["question_type"]
            paragraph = state.metadata.get("paragraph", "")

            # Get category-specific system prompt
            # Raises ValueError if invalid question_type or prompt_strategy
            system_prompt = get_system_prompt(question_type, prompt_strategy)

            # Start fresh conversation
            state.messages = []

            # System message with category-specific instructions
            system_msg = ChatMessageSystem(content=system_prompt)
            state.messages.append(system_msg)

            # Collect all responses
            responses = []

            # First question with image and optional paragraph context
            first_question_text = format_question_with_paragraph(
                questions[0], paragraph
            )

            user_msg_1 = ChatMessageUser(
                content=[
                    ContentImage(image=img_ref),
                    ContentText(text=first_question_text),
                ]
            )
            state.messages.append(user_msg_1)

            result_1 = await generate(state)
            responses.append(result_1.output.completion)

            # Remaining questions (text only, building on conversation context)
            # This is especially important for Conversational type where questions
            # reference previous answers using words like "this", "that", etc.
            for i in range(1, len(questions)):
                # No paragraph context for subsequent questions
                question_text = f"Question: {questions[i]}"

                user_msg = ChatMessageUser(content=question_text)
                state.messages.append(user_msg)

                result = await generate(state)
                responses.append(result.output.completion)

            # Get the final response (for Conversational, only last answer matters)
            # For other types, also use last answer (typically there's only one anyway)
            final_response = responses[-1]

            # Extract the final answer based on prompting strategy
            # This makes logs more readable and puts the clean answer in completion
            extracted_answer = extract_final_answer(final_response, prompt_strategy)

            # Store extracted answer in completion (for logging/display)
            state.output.completion = extracted_answer

            # Store all responses and full final response in metadata for debugging
            state.metadata["all_responses"] = responses
            state.metadata["full_response"] = final_response
            state.metadata["prompt_strategy"] = prompt_strategy

            return state

        except Exception as e:
            # Gracefully handle any errors (invalid question type, missing metadata, etc.)
            # Store error in metadata and return empty completion
            # This allows the evaluation to continue with remaining samples
            state.output.completion = ""
            state.metadata["solver_error"] = f"{type(e).__name__}: {str(e)}"
            return state

    return solve
