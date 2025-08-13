from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if state.messages[-1].content.strip().upper() == target.target[0]:  # type: ignore
            return Score(value=1.0)
        else:
            return Score(value=0.0)

    return score
