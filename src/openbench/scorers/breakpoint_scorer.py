"""
Breakpoint scorer for checking if agent successfully used submit_solution tool.
"""

from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def breakpoint_scorer() -> Scorer:
    """
    Score function implementations by checking submit_solution tool output.

    Scoring formula (per paper): Binary score for whether all tests pass

    Returns:
    - 1.0 if submit_solution reported all tests passed
    - 0.0 otherwise (tool not used, tests failed, or agent timed out)

    Note: This scorer only checks tool messages. The agent MUST use submit_solution()
    successfully to receive a non-zero score. There is no fallback evaluation.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Check if submit_solution succeeded by parsing messages
        # Look for success indicator in tool responses
        if hasattr(state, "messages") and state.messages:
            for msg in state.messages:
                # Check tool responses for success message
                if (
                    hasattr(msg, "role")
                    and msg.role == "tool"
                    and hasattr(msg, "content")
                    and msg.content
                    and "✅" in msg.content
                    and "SUCCESS! All tests passed!" in msg.content
                ):
                    return Score(
                        value=1.0,
                        answer=state.output.completion[:500]
                        if state.output and state.output.completion
                        else "",
                        explanation="Success! Agent's submit_solution passed all tests.",
                    )

        # No success message found - agent did not successfully use submit_solution()
        return Score(
            value=0.0,
            answer=state.output.completion[:500]
            if state.output and state.output.completion
            else "",
            explanation="Agent did not successfully submit a solution via submit_solution() tool, or tests did not pass.",
        )

    return score
