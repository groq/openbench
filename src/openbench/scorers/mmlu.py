from collections import defaultdict
import numpy as np
from inspect_ai.scorer import (
    Metric,
    Value,
    SampleScore,
    metric,
)
from openbench.scorers.mcq import mmlu_simple_eval_scorer

# Adapted from https://github.com/openai/simple-evals
SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


@metric
def category_accuracy_metrics() -> Metric:
    """
    Calculates accuracy and standard deviation for specific subject categories:
    stem, other, social_sciences, humanities.
    """

    def metric_calculator(scores: list[SampleScore]) -> Value:  # Value will be a dict
        # Define the categories we care about for reporting
        categories_to_report = ["stem", "other", "social_sciences", "humanities"]

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
                # Get the float value of the score (e.g., 1.0 for correct, 0.0 for incorrect)
                float_val = sample_score.score.as_float()
            except ValueError:
                # Log or handle if a score can't be converted, then skip it for these metrics
                print(
                    f"Warning: Could not convert score value '{sample_score.score.value}' "
                    f"to float for sample {sample_score.sample_id}. Skipping for category metrics."
                )
                continue  # Skip this sample_score for category calculations

            # Get subject and map to category
            if (
                sample_score.sample_metadata
                and "subject" in sample_score.sample_metadata
            ):
                subject = sample_score.sample_metadata["subject"]
                category = SUBJECT_TO_CATEGORY.get(subject)
                if (
                    category in categories_to_report
                ):  # Only collect for categories we're reporting
                    category_float_scores[category].append(float_val)

        # Calculate and populate per-category metrics in the results dictionary
        for cat_name in categories_to_report:
            cat_scores = category_float_scores[cat_name]
            if cat_scores:  # If there are any scores for this category
                results[cat_name] = float(np.mean(cat_scores))
                results[f"{cat_name}:std"] = float(np.std(cat_scores))
            # If no scores for a category, it keeps the default 0.0 values initialized earlier

        return results  # type: ignore

    return metric_calculator


# Re-export the scorer from mcq.py
# This keeps backward compatibility while using the unified scorer
__all__ = [
    "mmlu_simple_eval_scorer",
    "category_accuracy_metrics",
    "SUBJECT_TO_CATEGORY",
]
