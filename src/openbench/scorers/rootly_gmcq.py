from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        # Get the text content from the last message
        last_message = state.messages[-1]

        # Handle both string and list content types
        if isinstance(last_message.content, str):
            answer = last_message.content.strip().upper()
        elif isinstance(last_message.content, list):
            # Use the text property which properly concatenates text content
            answer = last_message.text.strip().upper()
        else:
            answer = ""

        if answer == target.target[0]:
            return Score(value=1.0)
        else:
            return Score(value=0.0)

    return score
