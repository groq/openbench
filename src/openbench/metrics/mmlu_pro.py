"""MMLU Pro metrics for evaluation scoring."""

from collections import defaultdict
from typing import List

import numpy as np
from inspect_ai.scorer import (
    Metric,
    Value,
    SampleScore,
    metric,
)


@metric
def category_accuracy_metrics() -> Metric:
    """
    Calculates accuracy and standard deviation for specific subject categories:
    math, health, physics, business, biology, chemistry, computer science, economics,
    engineering, philosophy, other, history, psychology, law.
    """

    def metric_calculator(scores: List[SampleScore]) -> Value:  # Value will be a dict
        categories_to_report = [
            "math",
            "health",
            "physics",
            "business",
            "biology",
            "chemistry",
            "computer science",
            "economics",
            "engineering",
            "philosophy",
            "other",
            "history",
            "psychology",
            "law",
        ]

        # Initialize results with default values for all expected metrics
        results = {}
        for cat_name in categories_to_report:
            results[cat_name] = 0.0
            results[f"{cat_name}:std"] = 0.0

        if not scores:
            return results  # type: ignore # Return defaults if no scores

        # Use defaultdict to easily collect scores per category
        category_float_scores = defaultdict(list)

        for sample_score in scores:
            try:
                sample_metadata = sample_score.sample_metadata
                if sample_metadata is not None:
                    category = sample_metadata["category"]
                    # Only collect for categories we're reporting
                    if category in categories_to_report:
                        # Get the float value of the score (e.g., 1.0 for correct, 0.0 for incorrect)
                        float_val = sample_score.score.as_float()
                        category_float_scores[category].append(float_val)
            except ValueError:
                # Log or handle if a score can't be converted, then skip it for these metrics
                print(
                    f"Warning: Could not convert score value '{sample_score.score.value}' "
                    f"to float for sample {sample_score.sample_id}. Skipping for category metrics."
                )
                continue  # Skip this sample_score for category calculations

        # Calculate and populate per-category metrics in the results dictionary
        for cat_name in categories_to_report:
            cat_scores = category_float_scores[cat_name]
            if cat_scores:  # If there are any scores for this category
                results[cat_name] = float(np.mean(cat_scores))
                results[f"{cat_name}:std"] = float(np.std(cat_scores))
            # If no scores for a category, it keeps the default 0.0 values initialized earlier

        return results  # type: ignore

    return metric_calculator
