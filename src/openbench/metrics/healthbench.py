"""HealthBench metrics for evaluation scoring."""

from collections import defaultdict

import numpy as np
from inspect_ai.scorer import metric


@metric
def healthbench_metrics():
    """Calculate HealthBench specific metrics including tag scores."""

    def metric_fn(scores):
        if not scores:
            return {}

        # Aggregate tag scores
        tag_values = defaultdict(list)
        for score in scores:
            if score.metadata:
                for key, value in score.metadata.items():
                    if key.startswith("tag_") and isinstance(value, (int, float)):
                        tag_values[key].append(value)

        # Calculate mean for each tag
        result = {}
        for key, values in tag_values.items():
            result[key] = float(np.clip(np.mean(values), 0, 1))

        return result

    return metric_fn
