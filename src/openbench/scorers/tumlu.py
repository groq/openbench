from openbench.scorers.mcq import tumlu_simple_eval_scorer

# Re-export the scorer from mcq.py
# This keeps backward compatibility while using the unified scorer
__all__ = [
    "tumlu_simple_eval_scorer",
]
