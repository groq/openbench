import re

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox

TIMEOUT = 3


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
@scorer(metrics=[accuracy(), stderr()])
def verify() -> Scorer:
    """
    Scorer for HumanEval tasks. Verifies the correctness of generated code
    by executing it against the provided test cases in a sandboxed environment.

    Returns:
        Scorer: The verification scorer function.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score a model's output by running the generated code and test cases.

        Args:
            state (TaskState): The current task state containing model output and metadata.
            target (Target): The target output (not used).

        Returns:
            Score: The result of the verification, including correctness and explanation.
        """
        answer = find_code(state.output.completion)
        code = [
            state.metadata["prompt"],
            answer,
            "\n",
            state.metadata["test"],
            "\n",
            f"check({state.metadata['entry_point']})",
        ]

        try:
            result = await sandbox().exec(
                cmd=["python", "-c", "".join(code)],
                timeout=TIMEOUT,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")

        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=answer,
            explanation="".join(
                ["The following verification code was executed:\n\n"]
                + ["```python\n\n"]
                + code
                + ["\n```"]
                + (
                    [f"\nThe submission was incorrect\n\n{result.stderr}"]
                    if not result.success
                    else [""]
                )
            ),
        )

    return score


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
def find_code(completion: str) -> str:
    """
    Extract code from a model completion, removing markdown and/or signature.

    Args:
        completion (str): The model's completion output.

    Returns:
        str: The extracted code.
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    if matches:
        extracted_answer = matches[0]
    else:
        extracted_answer = completion
    return str(extracted_answer)
