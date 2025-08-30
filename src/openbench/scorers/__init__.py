"""
Scorers module for bench evaluation framework.

This module provides various scoring functions for evaluating model outputs
across different types of tasks including mathematical problems, code generation,
and multiple choice questions.

Note: Eval-specific scorers (like those for MMLU and HumanEval) are not exported
from this module intentionally, as they are tailored to specific evaluation tasks
and should be imported directly from their respective modules when needed.
"""

from .score_last_number import score_last_integer, score_last_number
from .score_boxed import score_boxed
from .fallback_scorer import fallback_scorer
from .mcq import robust_mcq_scorer, extract_mcq_answer
from .robust_boxed import robust_boxed_scorer, aime_scorer, extract_boxed_answer
from .mmmu import mmmu_mixed_scorer

__all__ = [
    # Number scoring functions
    "score_last_integer",
    "score_last_number",
    # Boxed answer scoring
    "score_boxed",
    "robust_boxed_scorer",
    "aime_scorer",
    # Multiple choice scoring
    "robust_mcq_scorer",
    "extract_mcq_answer",
    # Utility functions
    "extract_boxed_answer",
    # Meta-scoring
    "fallback_scorer",
    # Mixed scoring for mcq and open-answer
    "mmmu_mixed_scorer",
]
