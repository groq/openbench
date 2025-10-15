"""MGSM metrics for evaluation scoring."""

from typing import Dict, List
from inspect_ai.scorer import (
    metric,
    Metric,
    Value,
)


@metric
def language_accuracy() -> Metric:
    """Calculate per-language accuracy metrics."""

    def metric_calculator(scores: list) -> Value:
        if not scores:
            return {}

        # Group scores by language
        language_scores: Dict[str, List[float]] = {}
        for sample_score in scores:
            metadata = sample_score.score.metadata
            if metadata and "language" in metadata:
                lang = metadata["language"]
                if lang not in language_scores:
                    language_scores[lang] = []
                language_scores[lang].append(sample_score.score.value)

        # Calculate accuracy per language
        metrics = {}
        for lang, lang_scores in language_scores.items():
            if lang_scores:
                accuracy = sum(lang_scores) / len(lang_scores)
                metrics[f"{lang}_accuracy"] = accuracy

        # Also calculate latin vs non-latin accuracy
        from openbench.datasets.mgsm import LATIN_LANGUAGES, NON_LATIN_LANGUAGES

        latin_scores = []
        non_latin_scores = []

        for sample_score in scores:
            metadata = sample_score.score.metadata
            if metadata and "language" in metadata:
                lang = metadata["language"]
                score_val = sample_score.score.value
                if lang in LATIN_LANGUAGES:
                    latin_scores.append(score_val)
                elif lang in NON_LATIN_LANGUAGES:
                    non_latin_scores.append(score_val)

        if latin_scores:
            metrics["latin_accuracy"] = sum(latin_scores) / len(latin_scores)
        if non_latin_scores:
            metrics["non_latin_accuracy"] = sum(non_latin_scores) / len(
                non_latin_scores
            )

        return metrics

    return metric_calculator
