"""DROP metrics for evaluation scoring."""

import re
import string
from typing import List, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from inspect_ai.scorer import (
    metric,
    Metric,
    Value,
    SampleScore,
)


# Answer extraction and normalization functions from simple-evals


def _remove_articles(text: str) -> str:
    """Remove articles from text."""
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    """Fix whitespace in text."""
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    """Remove punctuation from text unless it's a number."""
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def _tokenize(text: str) -> List[str]:
    """Tokenize text by spaces and hyphens."""
    return re.split(" |-", text)


def _is_number(text: str) -> bool:
    """Check if text represents a number."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    """Normalize a number to its float representation."""
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[List[str], List[Set[str]]]:
    """Convert answer(s) to normalized spans and token bags."""
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    """Check if numbers in gold and predicted bags match."""
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    """Compute F1 score between predicted and gold token bags."""
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    ) * 100
    return f1


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores.tolist()


def get_drop_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]],
    gold: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = float(np.mean(f1_per_bag))
    f1 = round(f1, 2)
    return exact_match, f1


@metric
def drop_metrics() -> Metric:
    """Calculate DROP specific metrics: F1 and exact match."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return {
                "exact_match": 0.0,
                "f1": 0.0,
            }

        total_em = 0.0
        total_f1 = 0.0

        for sample_score in scores:
            metadata = sample_score.score.metadata
            if metadata:
                total_em += metadata.get("exact_match", 0.0)
                total_f1 += metadata.get("f1", 0.0)

        n = len(scores)
        return {
            "exact_match": total_em / n if n > 0 else 0.0,
            "f1": total_f1 / n if n > 0 else 0.0,
        }

    return metric_calculator
