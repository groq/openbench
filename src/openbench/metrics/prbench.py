"""PRBench metrics for evaluation scoring."""

from collections import defaultdict

import numpy as np
from inspect_ai.scorer import metric


@metric
def prbench_metrics():
    """Calculate PRBench specific metrics including field (domain) scores."""

    def metric_fn(scores):
        if not scores:
            return {}

        # Aggregate field (domain) scores
        field_values = defaultdict(list)
        for score in scores:
            if score.metadata:
                # Collect field scores using mean_clipped
                field = score.metadata.get("field", "")
                if field and "mean_clipped" in score.metadata:
                    field_values[f"field_{field}"].append(
                        score.metadata["mean_clipped"]
                    )

        # Calculate mean for each field (as done in PRBench)
        result = {}
        for key, values in field_values.items():
            if values:
                result[key] = float(np.mean(values))

        return result

    return metric_fn
