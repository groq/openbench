"""
ProgressiveMCPBench scorer (LLM-as-Judge).

Implements an LLM-based scorer for ProgressiveMCPBench that checks the 'final_answer'
against the 'answers' list from the dataset, tolerating formatting/ordering differences.
"""

import json
import re
from typing import List

from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Scorer,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import (
    get_model,
    ChatMessageUser,
    Model,
)
from openbench.metrics.grouped import grouped
from openbench.utils.text import PROGRESSIVEMCPBENCH_GRADER_TEMPLATE


def _extract_final_answer(state: TaskState) -> str | None:
    """Extract the 'final_answer' field from the JSON output."""
    if not state.output or not state.output.completion:
        return None

    raw = state.output.completion.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to find the first {...} block
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None

    value = obj.get("final_answer")
    if value is None:
        return None

    return str(value).strip() or None


@metric
def progressivemcpbench_metrics() -> Metric:
    """Custom metrics including category breakdown."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        total_score = sum(float(s.score.value) for s in scores if isinstance(s.score.value, (int, float)))
        total_count = len(scores)

        category_stats: dict[str, dict[str, float]] = {}
        for s in scores:
            category = (
                str(s.score.metadata.get("category", "unknown"))
                if s.score.metadata
                else "unknown"
            )
            stats = category_stats.setdefault(
                category, {"total_score": 0.0, "count": 0}
            )
            stats["count"] += 1
            if isinstance(s.score.value, (int, float)):
                stats["total_score"] += float(s.score.value)

        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats["count"] > 0:
                category_accuracies[f"{category}_accuracy"] = (
                    stats["total_score"] / stats["count"]
                )

        return {
            "accuracy": total_score / total_count if total_count > 0 else 0.0,
            "total_count": total_count,
            **category_accuracies,
        }

    return metric_calculator


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        progressivemcpbench_metrics(),
        grouped(group_key="category", metric=[accuracy(), stderr()], all=False),
    ]
)
def progressivemcpbench_scorer(
    model: str = "groq/gpt-oss-120b",
) -> Scorer:
    """Scorer for ProgressiveMCPBench using LLM-as-a-judge.

    Args:
        model: The model to use for grading (default: groq/gpt-oss-120b).
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Expected answers from target.text, which we set to a list in the dataset
        expected_raw = target.target if target is not None else []
        if isinstance(expected_raw, str):
            expected_list: List[str] = [expected_raw]
        else:
            expected_list = list(expected_raw or [])

        expected_list = [e for e in (str(x).strip() for x in expected_list) if e]

        # Safety: we should have already filtered empty answers at dataset time
        if not expected_list:
            # Treat as ungraded / auto-skip
            return Score(
                value=0.0, answer="", metadata={"skipped_no_expected_answer": True}
            )

        model_answer = _extract_final_answer(state)
        if not model_answer:
            return Score(
                value=0.0,
                answer="",
                metadata={
                    "match_type": "no_answer",
                    "reason": "Failed to extract final_answer from output",
                    "expected_answers": expected_list,
                },
            )

        # Construct prompt for the LLM judge
        prompt = PROGRESSIVEMCPBENCH_GRADER_TEMPLATE.format(
            model_answer=model_answer,
            expected_answers="\n".join(f"- {a}" for a in expected_list),
        )

        try:
            response = await grader_model.generate([ChatMessageUser(content=prompt)])
            grade_text = response.completion.strip().upper()
            
            # Look for CORRECT or INCORRECT in the response
            if "CORRECT" in grade_text and "INCORRECT" not in grade_text:
                value = 1.0
                match_type = "correct"
            elif "INCORRECT" in grade_text:
                value = 0.0
                match_type = "incorrect"
            else:
                # Fallback if model output is ambiguous, treat as incorrect but log it
                value = 0.0
                match_type = "ambiguous_grade"

        except Exception as e:
            value = 0.0
            match_type = "grading_error"
            grade_text = str(e)

        category = (
            state.metadata.get("category", "unknown") if state.metadata else "unknown"
        )

        # Set grade letter for display
        if value >= 1.0:
            grade_letter = "A"
        elif value >= 0.5:
            grade_letter = "C"
        else:
            grade_letter = "F"

        return Score(
            value=value,
            answer=model_answer or "",
            metadata={
                "expected_answers": expected_list,
                "match_type": match_type,
                "grading_response": grade_text,
                "category": category,
                "grade_letter": grade_letter,
            },
        )

    return score
