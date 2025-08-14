from inspect_ai.solver import TaskState, Generate
from inspect_ai.scorer import Score, accuracy, stderr, scorer


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer():
    async def score(state: TaskState, generate: Generate):
        if (
            state.messages[-1].content.split()[-1].replace(".", "").lower()  # type: ignore
            == state.target.target[0].lower()
        ):
            return Score(value=1.0)
        else:
            return Score(value=0.0)

    return score
