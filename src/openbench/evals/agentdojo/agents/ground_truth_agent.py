from inspect_ai.model import ChatCompletionChoice, ModelOutput
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def ground_truth_agent() -> Solver:
    """
    A solver that executes the ground truth tool calls for a task.

    This agent retrieves the ground truth tool calls from the task metadata
    and executes them in sequence. It's useful for validating that the
    ground truth implementation actually solves the task.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the task from metadata
        user_task = state.metadata.get("user_task")
        injection_task = state.metadata.get("injection_task")
        pre_environment = state.metadata.get("pre_environment")

        if not user_task:
            raise ValueError("No user task found in metadata")

        if not pre_environment:
            raise ValueError("No pre_environment found in metadata")

        # Determine which task's ground truth to execute
        if injection_task:
            # For injection tasks, we execute the injection task's ground truth
            task = injection_task
        else:
            # For user tasks, we execute the user task's ground truth
            task = user_task

        # Get the ground truth tool calls
        ground_truth_calls = task.ground_truth(pre_environment)

        # Execute each tool call one by one
        for tool_call in ground_truth_calls:
            # Create an assistant message with a single tool call
            assistant_message = ChatMessageAssistant(
                content=f"Executing ground truth tool call: {tool_call.function}",
                tool_calls=[tool_call],
            )

            # Add the assistant message to the conversation
            state.messages.append(assistant_message)

            # Execute the tool call
            tool_results, _ = await execute_tools(
                [assistant_message],
                state.tools,
            )

            # Add tool results to messages
            state.messages.extend(tool_results)

        # Set up the output for scoring
        if hasattr(task, "GROUND_TRUTH_OUTPUT") and task.GROUND_TRUTH_OUTPUT:
            # Add final assistant message with ground truth output
            final_message = ChatMessageAssistant(content=task.GROUND_TRUTH_OUTPUT)
            state.messages.append(final_message)

            # Create ModelOutput for scoring
            state.output = ModelOutput(
                model="ground_truth",
                choices=[ChatCompletionChoice(message=final_message)],
            )

        return state

    return solve
