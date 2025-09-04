import re
import string
from collections import Counter
from typing import Callable
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState


WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}


def normalize_answer(s: str, question: str) -> str:
    """Normalize answer text for comparison."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def yesno(text):
        if "yes" == text[:3] or "no" == text[:2]:
            text = text.split()[0]
        return text

    def replace_text(text):
        return (
            text.replace("this is ", "")
            .replace("it is ", "")
            .replace("&", ",")
            .replace("and", ",")
            .replace("percent", "")
            .replace("organisation", "organization")
            .replace("because of", "")
            .replace("because", "")
            .replace("due to", "")
            .replace("hours", "hrs")
            .replace("minites", "min")
        )

    def word2number(text):
        words = text.split()
        return " ".join(
            [
                str(WORD_NUMBER_MAP[word]) if word in WORD_NUMBER_MAP else word
                for word in words
            ]
        )

    def remove_unit(text, question):
        if "how many" in question:
            idx = question.find("how many")
            unit = question[idx + len("how many") :].split()[0]
            text = text.replace(unit, "")
        if "which" in question:
            idx = question.find("which")
            unit = question[idx + len("which") :].split()[0]
            text = text.replace(unit, "")
        return text

    return word2number(
        white_space_fix(
            yesno(
                remove_articles(
                    remove_punc(remove_unit(replace_text(lower(s)), question))
                )
            )
        )
    )


def calculate_f1_em(
    prediction_tokens: list[str], ground_truth_tokens: list[str]
) -> tuple[float, float, float, bool]:
    """Calculate F1 score, precision, recall, and exact match."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0, False

    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    exact_match = prediction_tokens == ground_truth_tokens

    return f1, precision, recall, exact_match


@metric
def slide_vqa_metrics() -> Metric:
    """Calculate SlideVQA specific metrics including F1 scores."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {
                "f1": 0.0,
                "exact_match": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_em = 0

        for sample_score in scores:
            metadata = sample_score.score.metadata
            if metadata:
                total_f1 += metadata.get("f1", 0.0)
                total_precision += metadata.get("precision", 0.0)
                total_recall += metadata.get("recall", 0.0)
                total_em += 1 if metadata.get("exact_match", False) else 0

        count = len(scores)
        return {
            "f1": total_f1 / count,
            "exact_match": total_em / count,
            "precision": total_precision / count,
            "recall": total_recall / count,
        }

    return metric_calculator


@scorer(metrics=[slide_vqa_metrics()])
def slide_vqa_scorer() -> Callable:
    """SlideVQA scorer using F1 and exact match evaluation."""

    async def score(state: TaskState, target: Target) -> Score:
        # Get question from input
        question = state.input_text

        # Get the model's response
        model_response = state.output.completion

        # Get ground truth answer
        ground_truth = target.text

        # Normalize both prediction and ground truth
        prediction_normalized = normalize_answer(model_response, question)
        ground_truth_normalized = normalize_answer(ground_truth, question)

        # Tokenize
        prediction_tokens = prediction_normalized.split()
        ground_truth_tokens = ground_truth_normalized.split()

        # Calculate metrics
        f1, precision, recall, exact_match = calculate_f1_em(
            prediction_tokens, ground_truth_tokens
        )

        # Score is 1.0 for exact match, otherwise use F1 score
        score_value = 1.0 if exact_match else f1

        return Score(
            value=score_value,
            answer=model_response,
            explanation=f"F1: {f1:.3f}, EM: {exact_match}, Precision: {precision:.3f}, Recall: {recall:.3f}",
            metadata={
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "exact_match": exact_match,
                "prediction_normalized": prediction_normalized,
                "ground_truth_normalized": ground_truth_normalized,
                "qa_id": state.metadata.get("qa_id") if state.metadata else None,
            },
        )

    return score
