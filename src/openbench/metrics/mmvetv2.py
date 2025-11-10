"""MM-Vet v2 capability-specific metrics.

Provides individual metrics to evaluate performance across MM-Vet v2's
capability dimensions:
- rec: Recognition
- ocr: Optical Character Recognition
- know: Knowledge
- gen: Language Generation
- spat: Spatial Awareness
- math: Mathematics
- seq: Sequential Reasoning
"""

from collections import defaultdict
from typing import Any, List, cast

import numpy as np
from inspect_ai.scorer import Metric, SampleScore, Value, metric


# The 7 core capabilities in MM-Vet v2
CORE_CAPABILITIES = ["rec", "ocr", "know", "gen", "spat", "math", "seq"]


@metric
def mmvetv2_capability_metrics() -> Metric:
    """Compute MM-Vet v2 metrics with per-capability breakdown.

    Computes accuracy, stderr, and std for:
    - Overall performance across all samples
    - Each of 7 capability dimensions (rec, ocr, know, gen, spat, math, seq)

    Samples can have multiple capabilities and contribute to each capability's metrics.

    Returns:
        Metric function that computes overall and per-capability statistics
    """

    def compute(scores: List[SampleScore]) -> Value:
        results = {}

        # Extract all score values for overall metrics
        all_values = [cast(float, s.score.value) for s in scores]

        # Overall metrics
        if all_values:
            results["accuracy"] = float(np.mean(all_values))
            results["stderr"] = float(np.std(all_values) / np.sqrt(len(all_values)))
            results["std"] = float(np.std(all_values))

        # Group scores by capability (single pass, handles multi-label)
        by_capability: dict[str, List[float]] = defaultdict(list)

        for score in scores:
            capabilities = (
                score.sample_metadata.get("capability", [])
                if score.sample_metadata
                else []
            )

            # Add score value to each capability it belongs to
            if isinstance(capabilities, list):
                score_val = cast(float, score.score.value)
                for cap in capabilities:
                    if cap in CORE_CAPABILITIES:
                        by_capability[cap].append(score_val)

        # Per-capability metrics (only for capabilities with samples)
        for cap in CORE_CAPABILITIES:
            if cap in by_capability:
                cap_values = by_capability[cap]
                results[f"{cap}_accuracy"] = float(np.mean(cap_values))
                results[f"{cap}_stderr"] = float(
                    np.std(cap_values) / np.sqrt(len(cap_values))
                )
                results[f"{cap}_std"] = float(np.std(cap_values))

        return results  # type: ignore

    return compute


def mmvetv2_capability_metrics_list() -> List[Any]:
    """Return metrics as a list for Task configuration.

    Returns:
        List containing the mmvetv2_capability_metrics function
    """
    return [mmvetv2_capability_metrics()]
