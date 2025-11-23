"""
ProgressiveMCPBench scorer (Exact/Fuzzy Match).

Implements an exact and fuzzy string matching scorer for ProgressiveMCPBench,
checking against the 'answers' list from the dataset. No LLM-as-a-judge.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Callable, List, Any

from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState
from openbench.metrics.grouped import grouped


def _normalize(s: str) -> str:
    """Normalize string for comparison (lowercase, collapse whitespace)."""
    if not s:
        return ""
    s = str(s).strip().lower()
    # Collapse multiple spaces into one
    s = re.sub(r"\s+", " ", s)
    return s


def _similarity(a: str, b: str) -> float:
    """Calculate sequence matching ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


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
    """Custom metrics including category breakdown and partial (fuzzy) accuracy."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        correct_count = sum(1 for s in scores if s.score.value == 1.0)
        partial_count = sum(1 for s in scores if s.score.value == 0.5)
        total_count = len(scores)

        category_stats: dict[str, dict[str, int]] = {}
        for s in scores:
            category = (
                s.score.metadata.get("category", "unknown")
                if s.score.metadata
                else "unknown"
            )
            stats = category_stats.setdefault(
                category, {"correct": 0, "partial": 0, "total": 0}
            )
            stats["total"] += 1
            if s.score.value == 1.0:
                stats["correct"] += 1
            elif s.score.value == 0.5:
                stats["partial"] += 1

        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                category_accuracies[f"{category}_partial_or_better_accuracy"] = (
                    stats["correct"] + stats["partial"]
                ) / stats["total"]

        return {
            "correct_count": correct_count,
            "partial_count": partial_count,
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
    fuzzy_threshold: float = 0.85,
) -> Callable[[TaskState, Target], Score]:
    """Scorer for ProgressiveMCPBench using exact + fuzzy match on answers.
    
    Args:
        fuzzy_threshold: Minimum similarity ratio (0-1) to count as partial match (score 0.5).
    """

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
            return Score(value=0.0, answer="", metadata={"skipped_no_expected_answer": True})

        model_answer = _extract_final_answer(state)
        if not model_answer:
            value = 0.0
            match_type = "no_answer"
            best_similarity = 0.0
        else:
            norm_model = _normalize(model_answer)
            norm_expected = [_normalize(e) for e in expected_list]

            if norm_model in norm_expected:
                value = 1.0
                match_type = "exact"
                best_similarity = 1.0
            else:
                sims = [_similarity(norm_model, e) for e in norm_expected]
                best_similarity = max(sims) if sims else 0.0
                if best_similarity >= fuzzy_threshold:
                    value = 0.5
                    match_type = "fuzzy"
                else:
                    value = 0.0
                    match_type = "none"

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
                "best_similarity": best_similarity,
                "category": category,
                "grade_letter": grade_letter,
            },
        )

    return score
