from typing import Dict, List

from inspect_ai.scorer import SampleScore
from inspect_ai.scorer import metric


MIN_SCORE = 0.5


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
