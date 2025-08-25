"""MMMU (Massive Multi-discipline Multimodal Understanding) scorer."""

import re
from typing import Dict
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    stderr,
    scorer,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState


def extract_mmmu_answer(text: str) -> str:
    """Extract multiple choice answer (A, B, C, D) from model output."""
    if not text:
        return ""

    # Common patterns for extracting multiple choice answers
    patterns = [
        r"(?:answer|choice|option|select).*?([ABCD])\b",
        r"\b([ABCD])\)",
        r"\(([ABCD])\)",
        r"^([ABCD])(?:\.|:|\s|$)",
        r"\b([ABCD])(?:\.|:|\s|$)",
        r"(?:the )?answer is ([ABCD])",
        r"(?:i choose|i select) ([ABCD])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    letters = re.findall(r"\b([ABCD])\b", text.upper())
    if letters:
        return letters[0]

    return ""


@metric
def subject_accuracy() -> Metric:
    """Calculate accuracy per subject/subfield."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {}

        # Group scores by subfield
        subfield_scores: Dict[str, list[float]] = {}

        for sample_score in scores:
            metadata = sample_score.sample_metadata
            if isinstance(metadata, dict):
                subfield = metadata.get("subfield", "unknown")
            else:
                subfield = "unknown"
            if subfield not in subfield_scores:
                subfield_scores[subfield] = []
            # Ensure the score value is a float for calculation
            score_value = sample_score.score.value
            if isinstance(score_value, (int, float)):
                subfield_scores[subfield].append(float(score_value))
            else:
                # Skip non-numeric scores
                continue

        # Calculate accuracy per subfield
        result = {}
        for subfield, scores_list in subfield_scores.items():
            accuracy = sum(scores_list) / len(scores_list) if scores_list else 0.0
            result[f"accuracy_{subfield}"] = accuracy

        return result

    return metric_calculator


@metric
def difficulty_accuracy() -> Metric:
    """Calculate accuracy per difficulty level."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {}

        # Group scores by difficulty
        difficulty_scores: Dict[str, list[float]] = {}

        for sample_score in scores:
            # Ensure metadata is a dictionary before accessing it
            metadata = sample_score.sample_metadata
            if isinstance(metadata, dict):
                difficulty = metadata.get("topic_difficulty", "unknown")
            else:
                difficulty = "unknown"
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = []
            # Ensure the score value is a float for calculation
            score_value = sample_score.score.value
            if isinstance(score_value, (int, float)):
                difficulty_scores[difficulty].append(float(score_value))
            else:
                # Skip non-numeric scores
                continue

        # Calculate accuracy per difficulty
        result = {}
        for difficulty, scores_list in difficulty_scores.items():
            accuracy = sum(scores_list) / len(scores_list) if scores_list else 0.0
            result[f"accuracy_{difficulty.lower()}"] = accuracy

        return result

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), subject_accuracy(), difficulty_accuracy()])
def mmmu_scorer() -> Scorer:
    """MMMU scorer for multiple choice questions."""

    async def score(state: TaskState, target: Target) -> Score:
        extracted_answer = extract_mmmu_answer(state.output.completion)
        target_answer = target.text.strip().upper()

        # Check if extracted answer matches target
        is_correct = extracted_answer == target_answer

        # Get additional metadata for analysis
        metadata = state.metadata if isinstance(state.metadata, dict) else {}
        subfield = metadata.get("subfield", "")
        difficulty = metadata.get("topic_difficulty", "")
        num_images = metadata.get("num_images", 0)

        return Score(
            value=1.0 if is_correct else 0.0,
            answer=extracted_answer,
            metadata={
                "extracted_answer": extracted_answer,
                "target_answer": target_answer,
                "subfield": subfield,
                "difficulty": difficulty,
                "num_images": num_images,
                "is_correct": is_correct,
                "raw_output": state.output.completion,
            },
        )

    return score
