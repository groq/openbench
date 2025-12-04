"""
ProgressiveMCPBench scorer (LLM-as-Judge).

Implements an LLM-based scorer for ProgressiveMCPBench that checks the 'final_answer'
against the expected answer from the dataset, using the SimpleQA grading template.
"""

import json
import re

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
from openbench.scorers.simpleqa import GRADER_TEMPLATE


def _extract_final_answer(state: TaskState) -> str | None:
    """Extract the 'final_answer' field from the JSON output."""
    if not state.output or not state.output.completion:
        print(f"DEBUG: No output or completion in state")
        return None

    raw = state.output.completion.strip()
    print(f"DEBUG: Raw completion length: {len(raw)}")
    print(f"DEBUG: Raw completion preview: {repr(raw[:100])}")

    # Try direct JSON parse
    try:
        obj = json.loads(raw)
        print(f"DEBUG: Direct JSON parse succeeded, keys: {list(obj.keys()) if isinstance(obj, dict) else 'not dict'}")
    except json.JSONDecodeError as e:
        print(f"DEBUG: Direct JSON parse failed: {e}")
        # Fallback: try to find the first {...} block
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            print(f"DEBUG: No JSON block found")
            return None
        try:
            obj = json.loads(m.group(0))
            print(f"DEBUG: Fallback JSON parse succeeded")
        except json.JSONDecodeError as e2:
            print(f"DEBUG: Fallback JSON parse failed: {e2}")
            return None
    except Exception as e:
        print(f"DEBUG: Unexpected error in JSON parsing: {e}")
        return None

    if not isinstance(obj, dict):
        print(f"DEBUG: Parsed object is not a dict: {type(obj)}")
        return None

    value = obj.get("final_answer")
    if value is None:
        print(f"DEBUG: No 'final_answer' key in parsed object")
        return None

    result = str(value).strip() or None
    print(f"DEBUG: Extracted final_answer: {repr(result)}")
    return result


@metric
def progressivemcpbench_metrics() -> Metric:
    """Custom metrics including category breakdown."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        total_score = sum(
            float(s.score.value)
            for s in scores
            if isinstance(s.score.value, (int, float))
        )
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
    model: str = "groq/openai/gpt-oss-120b",
) -> Scorer:
    """Scorer for ProgressiveMCPBench using LLM-as-a-judge with SimpleQA grading.

    Args:
        model: The model to use for grading (default: groq/openai/gpt-oss-120b).
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the expected answer (now a single string)
        expected_answer = target.text if target else ""

        if not expected_answer:
            return Score(
                value=0.0, answer="", metadata={"skipped_no_expected_answer": True}
            )

        model_answer = _extract_final_answer(state)
        if not model_answer:
            return Score(
                value=0.0,
                answer="",
                metadata={
                    "grade": "not_attempted",
                    "grade_letter": "C",
                    "reason": "Failed to extract final_answer from output",
                    "expected_answer": expected_answer,
                },
            )

        # Get the question from state input
        question = state.input_text

        # Get scorer instructions from metadata if present
        scorer_instructions = (
            state.metadata.get("scorer_instructions") if state.metadata else None
        )

        # Build the gold target, optionally including scorer instructions
        gold_target = expected_answer
        if scorer_instructions:
            gold_target = f"{expected_answer}\n\nNote: {scorer_instructions}"

        # Use SimpleQA's grading template
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=gold_target,
            predicted_answer=model_answer,
        )

        try:
            response = await grader_model.generate(
                [ChatMessageUser(content=grader_prompt)]
            )
            grading_text = response.completion.strip()

            # Extract the grade letter (A, B, or C)
            match = re.search(r"(A|B|C)", grading_text)
            grade_letter = match.group(0) if match else "C"

            # Map letter to grade and score
            grade_map = {
                "A": ("correct", 1.0),
                "B": ("incorrect", 0.0),
                "C": ("not_attempted", 0.0),
            }

            grade_name, score_value = grade_map.get(
                grade_letter, ("not_attempted", 0.0)
            )

        except Exception as e:
            score_value = 0.0
            grade_name = "grading_error"
            grade_letter = "C"
            grading_text = str(e)

        category = (
            state.metadata.get("category", "unknown") if state.metadata else "unknown"
        )

        return Score(
            value=score_value,
            answer=model_answer or "",
            metadata={
                "expected_answer": expected_answer,
                "grade": grade_name,
                "grade_letter": grade_letter,
                "grading_response": grading_text,
                "category": category,
            },
        )

    return score
