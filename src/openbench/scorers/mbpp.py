import re
from inspect_ai.util import sandbox, ExecResult
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
    accuracy,
    stderr,
)
from inspect_ai.solver import TaskState

TIMEOUT = 3


def parse_mbpp_response(response: str) -> str:
    # get model code from between [BEGIN] and [DONE]
    if not response:
        return ""
    pattern = r"\[BEGIN\](.*?)\[DONE\]"
    match = re.search(pattern, response, re.DOTALL)

    return match.group(1).strip() if match else ""


@scorer(metrics=[accuracy(), stderr()])
def verify() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        """
        Score model MBPP output by running the generated code and test cases.

        Args:
            state (TaskState): The current task state containing model output and metadata.
            target (Target): The target output (not used).

        Returns:
            Score: Verification score, 1.0 if successful, 0.0 otherwise.
        """
        model_answer = parse_mbpp_response(state.output.completion)
        code = [
            state.metadata["test_setup_code"],
            "\n",
            model_answer,
            "\n",
            state.metadata["tests"],
        ]
        try:
            result = await sandbox().exec(
                cmd=["python", "-c", "".join(code)],
                timeout=TIMEOUT,
            )
        except Exception as e:
            result = ExecResult(False, 1, "", f"Verification error: {e}")
        return Score(
            value=1.0 if result.success else 0.0,
            answer=model_answer,
            explanation="The following verification code was executed:\n```python\n{code}\n```\n\n{result.stderr}".format(
                code="".join(code), result=result
            ),
        )

    return score
