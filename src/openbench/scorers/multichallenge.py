from __future__ import annotations
import json
import re
from typing import Any, Dict

from inspect_ai.model import get_model, ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    accuracy,
    stderr,
    Scorer,
)
from openbench.metrics.multichallenge import multichallenge_metrics

JUDGE_TEMPLATE = """
You are evaluating whether an assistant's response correctly answers a target question.
Be STRICT. Respond with a JSON object:

```json
{{"reasoning": "...", "verdict": "YES" or "NO"}}
```

<MODEL_RESPONSE>
{model_response}
</MODEL_RESPONSE>

<TARGET_QUESTION>
{target_question}
</TARGET_QUESTION>
""".strip()


def _parse_verdict(text: str) -> Dict[str, Any]:
    fenced = re.findall(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S)
    candidates = fenced + re.findall(r"({.*})", text, flags=re.S)

    for blob in candidates:
        try:
            obj = json.loads(blob)
            v = str(obj.get("verdict", "")).strip().upper()
            if v in {"YES", "NO"}:
                return {
                    "reasoning": str(obj.get("reasoning", "")).strip(),
                    "verdict": v,
                }
        except Exception:
            pass

    up = text.upper()
    if "YES" in up:
        return {"reasoning": text.strip(), "verdict": "YES"}
    if "NO" in up:
        return {"reasoning": text.strip(), "verdict": "NO"}

    return {"reasoning": text.strip(), "verdict": "NO"}


@scorer(metrics=[accuracy(), stderr(), multichallenge_metrics()])
def multichallenge_scorer(
    model: str = "openai/gpt-4.1-2025-04-14",
) -> Scorer:
    """
    MultiChallenge scorer.

    Uses a secondary "judge" model to evaluate free-form response to a
    target question. The judge model produces a structured verdict (PASS/FAIL)
    along with reasoning, which is parsed and compared against expected criteria.

    Args:
        model: Model identifier for the judging model used to evaluate responses.
               Defaults to `openai/gpt-4.1-2025-04-14`.

    Returns:
        Scorer function that executes the judge model, parses its verdict,
        and produces a Score with accuracy and diagnostic metadata.
    """
    model_instance = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata or {}
        question_id = md.get("question_id")
        axis = md.get("axis")
        target_question = str(md.get("target_question", ""))
        pass_criteria = str(md.get("pass_criteria", "")).strip().upper()

        candidate = state.output.completion if state.output else ""

        judge_prompt = JUDGE_TEMPLATE.format(
            model_response=candidate,
            target_question=target_question,
        )
        judge = await model_instance.generate([ChatMessageUser(content=judge_prompt)])
        judge_text = (judge.completion or "").strip()

        parsed = _parse_verdict(judge_text)
        verdict = parsed["verdict"]
        passed = verdict == pass_criteria
        value = 1.0 if passed else 0.0

        explanation = (
            f"Judge verdict: {verdict} | Expected: {pass_criteria}\n"
            f"Reasoning: {parsed.get('reasoning', '')[:2000]}"
        )

        return Score(
            value=value,
            explanation=explanation,
            metadata={
                "question_id": question_id,
                "axis": axis,
                "verdict": verdict,
                "expected": pass_criteria,
                "passed": passed,
                "target_question": target_question,
            },
        )

    return score
