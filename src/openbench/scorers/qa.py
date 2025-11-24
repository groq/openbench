"""Generalized QA scorer for extractive and open-domain QA tasks.

Provides exact match (EM) and F1 scoring.
Used by: NaturalQuestions, TriviaQA, SQuAD, CoQA, and similar QA benchmarks.
"""

import re
import string
from typing import Callable, List, Tuple

from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

EXCLUDE = set(string.punctuation)


def _is_number(text: str) -> bool:
    """Check if text represents a number."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text: str) -> str:
    """Remove articles from text."""
    return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)


def _white_space_fix(text: str) -> str:
    """Fix whitespace in text."""
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    """Remove punctuation from text unless it's a number."""
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    return text


def _normalize_number(text: str) -> str:
    """Normalize a number to its float representation."""
    if _is_number(text):
        return str(float(text))
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenize text by spaces and hyphens."""
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Normalize answer: lowercase, remove articles, punctuation, whitespace.

    Numbers are preserved and normalized to their float representation.
    """
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(token.lower())))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    return " ".join(parts).strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match after normalization."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    # Look for "Answer: " pattern
    match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", response)
    if match:
        return match.group(1).strip()
    # Fall back to last non-empty line
    for line in reversed(response.strip().split("\n")):
        if line.strip():
            return line.strip()
    return response.strip()


def score_qa(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    """Score prediction against multiple ground truths, return (max_em, max_f1)."""
    max_em = 0.0
    max_f1 = 0.0
    for gt in ground_truths:
        if gt:
            max_em = max(max_em, compute_exact_match(prediction, gt))
            max_f1 = max(max_f1, compute_f1(prediction, gt))
    return max_em, max_f1


@scorer(metrics=[accuracy(), stderr()])
def qa_scorer() -> Callable:
    """QA scorer using exact match and F1 metrics.

    Expects target to be pipe-separated answers (e.g. "answer1|answer2").
    Score is 1.0 for exact match, F1 otherwise.
    """

    async def score(state: TaskState, target: Target) -> Score:
        prediction = extract_answer(state.output.completion)
        ground_truths = [t.strip() for t in target.text.split("|") if t.strip()]

        em, f1 = score_qa(prediction, ground_truths)
        score_value = em if em == 1.0 else f1

        return Score(
            value=score_value,
            answer=prediction,
            metadata={
                "exact_match": em,
                "f1": f1,
                "prediction": prediction,
                "ground_truths": ground_truths,
            },
        )

    return score
