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

from inspect_ai.scorer import accuracy, stderr, std
from inspect_ai.scorer._metric import (
    Metric,
    MetricProtocol,
    SampleScore,
    Value,
    metric,
    registry_info,
)


# The 7 core capabilities in MM-Vet v2
CORE_CAPABILITIES = ["rec", "ocr", "know", "gen", "spat", "math", "seq"]


@metric
def mmvetv2_capability_breakdown(
    base_metrics: List[Metric],
) -> Metric:
    """Create capability-specific breakdown metrics for MM-Vet v2.

    Args:
        base_metrics: List of base metrics to apply (e.g., [accuracy(), stderr(), std()])

    Returns:
        A metric that computes per-capability breakdown for each base metric
    """

    def capability_metric(scores: List[SampleScore]) -> Value:
        # Extract metric short names
        metric_names = [registry_info(m).name for m in base_metrics]
        short_names = [name.split("/")[-1] for name in metric_names]

        results: dict[str, Value] = {}

        # Compute overall metrics
        for m, short_name in zip(base_metrics, short_names):
            results[short_name] = cast(MetricProtocol, m)(scores)

        # Group scores by capability in a single pass
        by_capability: dict[str, List[SampleScore]] = defaultdict(list)

        for score in scores:
            # Safely get capabilities list from metadata
            capabilities = (
                score.sample_metadata.get("capability", [])
                if score.sample_metadata
                else []
            )

            # Add score to each capability it belongs to
            if isinstance(capabilities, list):
                for cap in capabilities:
                    if cap in CORE_CAPABILITIES:
                        by_capability[cap].append(score)

        # Compute per-capability metrics (only for capabilities with samples)
        for cap in CORE_CAPABILITIES:
            if cap in by_capability:
                for m, short_name in zip(base_metrics, short_names):
                    key = f"{cap}_{short_name}"
                    results[key] = cast(MetricProtocol, m)(by_capability[cap])

        return cast(Value, results)

    return capability_metric


def mmvetv2_capability_metrics() -> List[Any]:
    """Create capability-specific metrics for MM-Vet v2.

    Returns metrics for overall performance and individual performance
    on each of the 7 core capabilities. Since samples can have multiple
    capabilities, a sample contributes to the metric for each capability
    it contains.

    Returns:
        List of metric functions including overall and per-capability metrics
    """
    return [
        mmvetv2_capability_breakdown([accuracy(), stderr(), std()]),
    ]
