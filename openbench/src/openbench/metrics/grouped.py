from typing import Callable, Literal, Sequence, cast, Optional

import numpy as np

from inspect_ai.scorer._metric import (
    Metric,
    MetricProtocol,
    SampleScore,
    Value,
    ValueToFloat,
    metric,
    value_to_float,
    registry_info,
)


# Forked from the grouped metric in https://github.com/UKGovernmentBEIS/inspect_ai
@metric
def grouped(
    metric: Metric | Sequence[Metric],
    group_key: str,
    *,
    all: Literal["samples", "groups"] | Literal[False] = "samples",
    all_label: str = "all",
    value_to_float: ValueToFloat = value_to_float(),
    group_namer: Optional[Callable[[str, str], str]] = None,
) -> Metric:
    """
    Create a grouped metric that applies the given metric(s) to subgroups of samples.

    This function groups samples based on a metadata key and applies one or more metrics
    to each group. Optionally, it computes an aggregate score across all samples or all
    group-level scores.

    Args:
        metric: A metric or list of metrics to apply to each group of samples.
        group_key: Metadata key used to group samples.
        all: Determines whether and how to compute an aggregate "all" score.
            - "samples": Apply the metric(s) to all samples regardless of groups.
            - "groups": Calculate the mean of all group scores.
            - False: Do not compute an aggregate score.
        all_label: Label to use for the aggregate score when a single metric is used.
        value_to_float: Function to convert metric values to floats (used for averaging group scores).
        group_namer: Optional function to generate group-specific metric names; receives (group, metric_name).

    Returns:
        Metric: A new metric function returning a dictionary mapping group names (and optionally
        an "all" aggregate key) to their respective scores.
    """

    def grouped_metric(scores: list[SampleScore]) -> Value:
        # Normalize to list of metrics
        metrics = metric if isinstance(metric, (list, tuple)) else [metric]
        metric_names = [registry_info(m).name for m in metrics]
        short_names = [name.split("/")[-1] for name in metric_names]

        # Use default group_namer if none provided
        nonlocal group_namer, all_label
        if group_namer is None:

            def default_group_namer(group: str, metric_name: str) -> str:
                return f"{group}_{metric_name}"

            group_namer = default_group_namer

        # If only one metric and user didn't override label, use that metric's name
        if all_label == "all" and len(metrics) == 1:
            all_label = short_names[0]

        # Build map of group name → list of sample scores
        scores_by_group: dict[str, list[SampleScore]] = {}
        for score in scores:
            if score.sample_metadata is None or group_key not in score.sample_metadata:
                raise ValueError(
                    f"Sample {score.sample_id} has no '{group_key}' in metadata. "
                    "All samples must include this key to compute grouped metrics."
                )
            group_name = str(score.sample_metadata[group_key])
            scores_by_group.setdefault(group_name, []).append(score)

        # If requested, compute aggregate metric over all samples before group metrics
        all_metrics: dict[str, Value] = {}
        if all == "samples":
            for m, short_name in zip(metrics, short_names):
                key = all_label if len(metrics) == 1 else short_name
                all_metrics[key] = cast(MetricProtocol, m)(scores)

        # Compute metric for each group
        grouped_scores: dict[str, Value] = {}
        for group_name, group_scores in scores_by_group.items():
            for m, short_name in zip(metrics, short_names):
                key = group_namer(group_name, short_name)
                grouped_scores[key] = cast(MetricProtocol, m)(group_scores)

        # If requested, compute aggregate metric from group scores
        if all == "groups":
            for m, short_name in zip(metrics, short_names):
                key = all_label if len(metrics) == 1 else short_name
                group_keys = [group_namer(g, short_name) for g in scores_by_group]
                values = [value_to_float(grouped_scores[k]) for k in group_keys]
                all_metrics[key] = float(np.mean(values)) if values else 0.0

        # Return combined results
        if all is False:
            return cast(Value, grouped_scores)
        return cast(Value, {**all_metrics, **grouped_scores})

    return grouped_metric
