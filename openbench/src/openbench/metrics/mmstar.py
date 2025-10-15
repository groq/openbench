"""Metrics for MMStar evaluation.

Implements aggregate metrics aligned with the MMStar paper, including
multi-modal gain (MG) and multi-modal leakage (ML).
"""

from __future__ import annotations

from typing import List, DefaultDict, Optional, Dict
from collections import defaultdict
from inspect_ai.scorer import Metric, SampleScore, Value, metric


@metric
def mmstar_metrics() -> Metric:
    """Compute MMStar aggregate metrics (Sv, Swv, St, MG, ML)."""

    def compute(scores: List[SampleScore]) -> Value:
        # structure for per-category metrics dict
        metrics: DefaultDict[str, Dict[str, float | int]] = defaultdict(
            lambda: {
                "with_vision": 0.0,
                "without_vision": 0.0,
                "text_base": 0.0,
                "total": 0,
            }
        )

        # iterate over score, add to per-category metrics dict
        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            category: str = metadata.get("category") or "unknown"
            metrics[category]["total"] += 1

            if metadata.get("with_vision_score") == 1.0:
                metrics[category]["with_vision"] += 1

            if metadata.get("without_vision_score") == 1.0:
                metrics[category]["without_vision"] += 1

            if metadata.get("base_model") and metadata.get("text_base_score") == 1.0:
                metrics[category]["text_base"] += 1

        # total number of samples = sum of catgeory totals
        total = sum(list(metrics[category]["total"] for category in metrics))

        # global accuracy metrics (Sv, Swv, St)
        sv = sum(list(metrics[category]["with_vision"] for category in metrics)) / total
        swv = (
            sum(list(metrics[category]["without_vision"] for category in metrics))
            / total
        )
        st = (
            sum(list(metrics[category]["text_base"] for category in metrics)) / total
            if metadata.get("base_model")
            else None
        )

        # multi-modal gain (MG)
        mg = sv - swv

        # multi-modal leakage (ML)
        ml = (
            max(0.0, swv - st) if st else None
        )  # if no base model, no multi-modal leakage

        # per-category Sv accuracy
        per_category_accuracy: Dict[str, Optional[float]] = {}
        for category, values in metrics.items():
            per_category_accuracy[category] = (
                values["with_vision"] / values["total"] if values["total"] > 0 else None
            )

        aggregate_scores = {
            "visual_score(sv)": sv,
            "without_visual_score(swv)": swv,
            "text_base_score(st)": st,
            # multi-modal gain
            "multi-modal_gain(mg)": mg,
            # multi-modal leakage
            "multi-modal_leakage(ml)": ml,
            # unpack per-category accuracy
            **per_category_accuracy,
        }

        return aggregate_scores

    return compute
