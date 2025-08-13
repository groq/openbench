from inspect_ai.solver import TaskState, Generate
from inspect_ai.scorer import scorer, Score, accuracy, stderr
from inspect_ai.util import ExecResult, sandbox


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer(SCENARIO: str, VERIFY_TIMEOUT: int = 30):
    async def score(state: TaskState, generate: Generate):
        if SCENARIO in ["codegeneration", "selfrepair", "codeexecution"]:
            try:
                result = await sandbox().exec(
                    cmd=["python", "-c", "".join(state.metadata["generated_code"])],
                    timeout=VERIFY_TIMEOUT,
                )
                # print(state.metadata["generated_code"])
                # print(result)
                if result.returncode == 0:
                    return Score(value=1.0)
                else:
                    return Score(value=0.0)
            except TimeoutError:
                result = ExecResult(False, 1, "", "Verification timed out.")

                return Score(value=0.0)
        else:
            return ValueError(f"Invalid scenario: {SCENARIO}")

    return score


@scorer(metrics=[accuracy(), stderr()])
def test_output_prediction_scorer(SCENARIO: str):
    async def score(state: TaskState, generate: Generate):
        if SCENARIO == "testoutputprediction":
            formatted_test_output = (
                state.metadata["predicted_test_output"].split("==")[-1].strip()
            )

            if (
                formatted_test_output.strip()
                == str(state.metadata["test"][0]["output"]).strip()
            ):
                return Score(value=1.0)
            else:
                return Score(value=0.0)
        else:
            return ValueError(f"Invalid scenario: {SCENARIO}")

    return score
