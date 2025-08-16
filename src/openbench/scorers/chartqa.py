from inspect_ai.scorer import accuracy, stderr, scorer
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target, Score


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer():
    async def score(state: TaskState, target: Target):
        output = state.output
        model_prediction = (
            output.choices[-1].message.content.split("Final Answer: ")[-1].strip()  # type: ignore
        )
        target_value = target.target[0].strip()
        if model_prediction == target_value:
            return Score(value=1.0)
        else:
            return Score(value=0.0)

    return score
