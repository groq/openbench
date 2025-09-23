from openbench.scorers.mcq import mmlu_simple_eval_scorer
from openbench.metrics.mmlu import category_accuracy_metrics, SUBJECT_TO_CATEGORY

# Re-export the scorer from mcq.py
# This keeps backward compatibility while using the unified scorer


# Re-export the scorer from mcq.py
# This keeps backward compatibility while using the unified scorer
__all__ = [
    "mmlu_simple_eval_scorer",
    "category_accuracy_metrics",
    "SUBJECT_TO_CATEGORY",
]
