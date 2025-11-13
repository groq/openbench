"""DocVQA scorer implementing ANLS (Average Normalized Levenshtein Similarity).

The ANLS metric is the standard evaluation metric for Document VQA tasks.
It uses normalized Levenshtein distance with a threshold to handle OCR errors gracefully.

Reference: https://arxiv.org/abs/1907.00490 (original ANLS paper)
"""

from typing import Any, Dict, List

import Levenshtein  # type: ignore[import-not-found]
from inspect_ai.scorer import Score, Scorer, Target, scorer


def normalize_answer(text: str) -> str:
    """Normalize answer text following original DocVQA evaluation.

    The original DocVQA implementation only applies case-insensitive matching.
    This stays faithful to the paper to ensure comparable leaderboard scores.

    Args:
        text: Answer text to normalize

    Returns:
        Normalized text (lowercase and whitespace-trimmed)
    """
    # Case-insensitive matching only (as per original DocVQA)
    return text.lower().strip()


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity between two strings.

    The similarity is computed as:
        similarity = 1 - (levenshtein_distance / max_length)

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    distance = Levenshtein.distance(s1, s2)
    similarity = 1.0 - (distance / max_len)
    return similarity


def anls_score_single(
    predicted: str,
    ground_truths: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute ANLS score for a single prediction against multiple ground truths.

    ANLS takes the maximum similarity across all ground truth answers and applies
    a threshold. Scores below the threshold are set to 0.

    Formula:
        s(answer, prediction) = 1 - NL(answer, prediction) if NL < threshold
                              = 0                         if NL >= threshold

        where NL = Normalized Levenshtein Distance

    Args:
        predicted: Model's predicted answer
        ground_truths: List of acceptable ground truth answers
        threshold: Threshold for accepting answers (default 0.5 as per DocVQA standard)

    Returns:
        Dictionary containing:
        - anls: Final ANLS score (0.0 to 1.0)
        - best_similarity: Similarity with best matching ground truth
        - best_match: The ground truth answer with highest similarity
        - is_exact_match: Whether prediction exactly matches any ground truth
    """
    # Handle empty cases
    if not ground_truths:
        return {
            "anls": 0.0,
            "best_similarity": 0.0,
            "best_match": "",
            "is_exact_match": False,
        }

    # Normalize strings: case-insensitive matching only (per original DocVQA)
    # The dataset provides multiple acceptable answers for variations
    predicted_normalized = normalize_answer(predicted)

    # Compute similarity with each ground truth and take maximum
    best_similarity = 0.0
    best_match = ground_truths[0]

    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        similarity = normalized_levenshtein_similarity(
            predicted_normalized, gt_normalized
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = gt

    # Apply threshold: if normalized Levenshtein distance >= threshold, score is 0
    # Since similarity = 1 - distance, we check if similarity < (1 - threshold)
    normalized_distance = 1.0 - best_similarity
    if normalized_distance >= threshold:
        anls = 0.0
    else:
        anls = best_similarity

    # Check for exact match (after normalization)
    is_exact_match = any(
        predicted_normalized == normalize_answer(gt) for gt in ground_truths
    )

    return {
        "anls": anls,
        "best_similarity": best_similarity,
        "best_match": best_match,
        "is_exact_match": is_exact_match,
    }


@scorer(metrics=[])
def docvqa_anls() -> Scorer:
    """Scorer for DocVQA using ANLS (Average Normalized Levenshtein Similarity).

    This scorer:
    1. Extracts the predicted answer from model output
    2. Computes ANLS score against all ground truth answers
    3. Takes maximum similarity across ground truths
    4. Applies threshold of 0.5 (scores below threshold become 0)
    5. Returns Score with value 'C' (correct) if ANLS > 0, 'I' (incorrect) otherwise

    The ANLS metric gracefully handles OCR errors by using edit distance rather than
    exact match, but still requires reasonably close answers via the threshold.

    Returns:
        Scorer function compatible with Inspect AI evaluation framework
    """

    async def score(state: Any, target: Target) -> Score:
        """Score a single model output against ground truth answers.

        Args:
            state: TaskState containing model output and metadata
            target: Target containing ground truth answer(s)

        Returns:
            Score object with value 'C' or 'I' and ANLS score in metadata
        """
        # Extract predicted answer from model output
        predicted = state.output.completion.strip() if state.output.completion else ""

        # Get ground truth answers from metadata
        # The metadata should contain the 'answers' field with all acceptable answers
        ground_truths = state.metadata.get("answers", [])

        # Fallback to target if answers not in metadata
        if not ground_truths and target.text:
            ground_truths = [target.text]

        # Compute ANLS score
        result = anls_score_single(
            predicted=predicted,
            ground_truths=ground_truths,
            threshold=0.5,
        )

        # Use the actual ANLS score as the value (0.0 to 1.0)
        # This is a continuous metric, not binary correct/incorrect
        value = result["anls"]

        # Store detailed scoring information in metadata
        score_metadata = {
            "anls": result["anls"],
            "best_similarity": result["best_similarity"],
            "best_match": result["best_match"],
            "is_exact_match": result["is_exact_match"],
            "predicted": predicted,
            "ground_truths": ground_truths,
            "question_types": state.metadata.get("question_types", []),
        }

        return Score(
            value=value,
            answer=predicted,
            explanation=f"ANLS: {result['anls']:.4f} (best match: '{result['best_match']}')",
            metadata=score_metadata,
        )

    return score
