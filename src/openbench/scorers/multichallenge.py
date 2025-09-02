from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Callable

from inspect_ai.model import get_model, ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    accuracy,
    stderr,
    metric,
    SampleScore,
    Scorer,
)

MIN_SCORE = 0.5

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


@metric
def multichallenge_metrics():
    """
    Aggregate per-axis pass rates for MultiChallenge tasks.

    Groups scores by (axis, question_id) and marks a question as "passed"
    on an axis if it passed at least once. Then computes:

      * axis_<axis>: fraction of passed questions for each axis
      * overall_multichallenge: average across all axes
    """

    def metric_fn(scores: List[SampleScore]) -> Dict[str, float]:
        from collections import defaultdict

        # use defaultdict for auto creating keys upon accessing, initialize with empty dict
        # structure: {axis: {qid: passed, ...}, ...}
        grouped_by_axis: Dict[str, Dict[str, bool]] = defaultdict(dict)

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            axis = metadata.get("axis")
            qid = metadata.get("question_id")
            try:
                float_val = sample_score.score.as_float()
            except ValueError:
                # Log or handle if a score can't be converted, then skip it for these metrics
                print(
                    f"Warning: Could not convert score value '{sample_score.score.value}' "
                    f"to float for sample {sample_score.sample_id}. Skipping for category metrics."
                )
                continue
            passed = bool(metadata.get("passed", float_val >= MIN_SCORE))
            if not axis or not qid:
                continue
            grouped_by_axis[axis][qid] = grouped_by_axis[axis].get(qid, False) or passed

        axis_rates: Dict[str, float] = {}
        for axis, per_q in grouped_by_axis.items():
            if not per_q:
                continue
            wins = sum(1 for ok in per_q.values() if ok)
            axis_rates[axis] = wins / len(per_q)

        overall = sum(axis_rates.values()) / len(axis_rates) if axis_rates else 0.0
        out = {f"axis_{k}": v for k, v in axis_rates.items()}
        out["overall_multichallenge"] = overall
        return out

    return metric_fn


@scorer(metrics=[accuracy(), stderr(), multichallenge_metrics()])
def multichallenge_scorer(
    model: str = "openai/o3-mini-2025-01-31",
) -> Scorer:
    """
    MultiChallenge scorer.

    Uses a secondary "judge" model to evaluate free-form response to a
    target question. The judge model produces a structured verdict (PASS/FAIL)
    along with reasoning, which is parsed and compared against expected criteria.

    Args:
        model: Model identifier for the judging model used to evaluate responses.
               Defaults to `openai/o3-mini-2025-01-31`.

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
