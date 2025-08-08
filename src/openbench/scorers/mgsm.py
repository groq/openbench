"""MGSM scorer for evaluating math problem solutions."""

from typing import Callable, Dict, List
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    metric,
    Metric,
    Value,
)
from inspect_ai.solver import TaskState
from openbench.utils.text import parse_numeric_answer, normalize_number


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


@scorer(metrics=[accuracy(), stderr(), language_accuracy()])
def mgsm_scorer() -> Callable:
    """MGSM scorer for evaluating math problem solutions."""

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's response
        model_output = state.output.completion

        # Get metadata from the sample
        metadata = state.metadata
        answer_prefix = metadata.get("answer_prefix", "Answer")
        language = metadata.get("language", "en")

        # Extract answer from model output
        extracted_answer = parse_numeric_answer(model_output, answer_prefix)

        # Normalize both extracted answer and target for comparison
        normalized_extracted = normalize_number(extracted_answer)
        normalized_target = normalize_number(target.text)

        # Score is 1.0 if they match, 0.0 otherwise
        is_correct = normalized_extracted == normalized_target
        score_value = 1.0 if is_correct else 0.0

        return Score(
            value=score_value,
            answer=extracted_answer if extracted_answer else "[No answer found]",
            explanation=f"Extracted: {extracted_answer}, Target: {target.text}, Normalized match: {is_correct}",
            metadata={
                "language": language,
                "extracted_answer": extracted_answer,
                "normalized_extracted": normalized_extracted,
                "normalized_target": normalized_target,
            },
        )

    return score
