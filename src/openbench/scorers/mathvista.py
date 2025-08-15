from inspect_ai.scorer import scorer, Target, Score
from inspect_ai.solver import TaskState
from inspect_ai.scorer import accuracy, stderr


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer():
    async def score(state: TaskState, target: Target):
        # assume non-reasoning model first
        model_pred = state.messages[-1].content
        # if model_pred is a list, it means the model is a reasoning model
        if type(model_pred) is list:
            model_pred = model_pred[-1].text  # type: ignore

        if "\\boxed{" in model_pred:
            model_pred = (
                model_pred.split("\\boxed{")[-1].strip().replace("}", "").strip()  # type: ignore
            )
            model_pred = model_pred.replace("*", "")
            model_pred = model_pred.replace("$", "")
        else:
            model_pred = model_pred.strip()  # type: ignore
            if model_pred[-1] == ".":
                model_pred = model_pred[:-1]
            model_pred = model_pred.split()[-1].strip()
        state.metadata["model_pred"] = model_pred
        if state.metadata["question_type"] == "multi_choice":
            try:
                char_to_int = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
                char = model_pred[0]
                char = char.replace(".", "")
                char = char.replace("*", "")
                model_pred_index = char_to_int[char]
                model_pred_value = state.choices[model_pred_index].value

                if 0 <= model_pred_index < len(state.choices):
                    model_pred_value = state.choices[model_pred_index].value
                else:
                    return Score(value=0.0)

                state.metadata["model_pred_value"] = model_pred_value
                state.metadata["target_value"] = target.target[0].strip()
                state.metadata["model_pred"] = str(
                    model_pred_value == target.target[0].strip()
                )

                if model_pred_value == target.target[0].strip():
                    return Score(value=1.0)
                else:
                    return Score(value=0.0)
            except KeyError:
                return Score(value=0.0)

        else:
            if model_pred == target.target[0].strip():
                return Score(value=1.0)
            else:
                return Score(value=0.0)

    return score
