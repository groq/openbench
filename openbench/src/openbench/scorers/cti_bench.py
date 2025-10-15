"""CTI-Bench scorers for cybersecurity benchmarking tasks."""

import re
from typing import Callable, Set
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target
from inspect_ai.solver import TaskState
from openbench.metrics.cti_bench import (
    technique_precision,
    technique_recall,
    technique_f1,
    exact_match_accuracy,
    mean_absolute_deviation,
    accuracy_within_threshold,
)


# ATE (ATT&CK Technique Extraction) Functions
def extract_technique_ids(text: str) -> Set[str]:
    """Extract MITRE ATT&CK technique IDs from model output."""
    if not text:
        return set()

    technique_ids = set()
    text_upper = text.upper()

    # Single comprehensive pattern for all T-ID formats
    all_patterns = [
        r"\bT\d{4}(?:\.\d{3})?\b",  # Basic T1234 or T1234.001
        r"(?:technique\s+)?(T\d{4})(?:\.\d{3})?(?:\s*[:\-,.]|\s|$)",  # Context patterns
    ]

    # Extract from all patterns
    for pattern in all_patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        for match in matches:
            # Extract main technique ID (remove subtechnique if present)
            main_technique = match.split(".")[0]
            technique_ids.add(main_technique)

    # Special handling for final line with only technique IDs
    lines = text.strip().split("\n")
    if lines:
        last_line = lines[-1].strip().upper()
        if re.match(r"^[T\d,\s\.]+$", last_line):
            final_matches = re.findall(r"T\d{4}(?:\.\d{3})?", last_line)
            technique_ids.update(match.split(".")[0] for match in final_matches)

    return technique_ids


def parse_ground_truth(gt_text: str) -> Set[str]:
    """Parse ground truth technique IDs from comma-separated string."""
    if not gt_text:
        return set()

    return {
        technique_id.strip().upper().split(".")[0]
        for technique_id in gt_text.split(",")
        if technique_id.strip() and technique_id.strip().upper().startswith("T")
    }


@scorer(
    metrics=[
        exact_match_accuracy(),
        technique_precision(),
        technique_recall(),
        technique_f1(),
        stderr(),
    ]
)
def cti_bench_ate_scorer() -> Callable:
    """Scorer for CTI-Bench ATE (ATT&CK Technique Extraction) task."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract technique IDs from model response
        predicted_techniques = extract_technique_ids(state.output.completion)
        ground_truth_techniques = parse_ground_truth(target.text.strip())

        # Calculate exact match
        is_exact_match = predicted_techniques == ground_truth_techniques

        # Calculate individual sample metrics for metadata
        if not predicted_techniques and not ground_truth_techniques:
            precision = recall = f1 = 1.0  # Perfect match when both are empty
        elif not predicted_techniques or not ground_truth_techniques:
            precision = recall = f1 = 0.0  # One is empty, the other is not
        else:
            tp = len(predicted_techniques & ground_truth_techniques)
            precision = tp / len(predicted_techniques) if predicted_techniques else 0.0
            recall = (
                tp / len(ground_truth_techniques) if ground_truth_techniques else 0.0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return Score(
            value=1.0 if is_exact_match else 0.0,
            answer=", ".join(sorted(predicted_techniques))
            if predicted_techniques
            else "None",
            metadata={
                "predicted_techniques": list(predicted_techniques),
                "ground_truth_techniques": list(ground_truth_techniques),
                "sample_precision": round(precision, 4),
                "sample_recall": round(recall, 4),
                "sample_f1": round(f1, 4),
                "raw_output": state.output.completion,
            },
        )

    return score


# MCQ (Multiple Choice Questions) Functions
def extract_multiple_choice_answer(text: str) -> str:
    """Extract multiple choice answer from model output."""
    if not text:
        return ""

    # Try various patterns to extract the answer
    patterns = [
        r"(?:answer|choice|option|select).*?([ABCD])\b",  # "answer is A", "choice B", etc.
        r"\b([ABCD])\)",  # "A)", "B)", etc.
        r"\(([ABCD])\)",  # "(A)", "(B)", etc.
        r"^([ABCD])(?:\.|:|\s|$)",  # Answer starts with letter
        r"\b([ABCD])(?:\.|:|\s|$)",  # Letter at word boundary
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: look for any A, B, C, or D in the text
    letters = re.findall(r"[ABCD]", text.upper())
    if letters:
        return letters[0]

    return ""


@scorer(metrics=[accuracy(), stderr()])
def cti_bench_mcq_scorer() -> Callable:
    """Scorer for CTI-Bench multiple choice questions."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract the answer from model response
        extracted_answer = extract_multiple_choice_answer(state.output.completion)
        target_answer = target.text.strip().upper()

        # Check if extracted answer matches target
        is_correct = extracted_answer == target_answer

        return Score(
            value=1.0 if is_correct else 0.0,
            answer=extracted_answer,
            metadata={
                "extracted_answer": extracted_answer,
                "target_answer": target_answer,
                "raw_output": state.output.completion,
            },
        )

    return score


# RCM (CVE→CWE vulnerability mapping) Functions
def extract_cwe_id(text: str) -> str:
    """Extract CWE ID from model output."""
    if not text:
        return ""

    # Try to find CWE-XXX pattern
    cwe_pattern = r"CWE-(\d+)"
    match = re.search(cwe_pattern, text, re.IGNORECASE)
    if match:
        return f"CWE-{match.group(1)}"

    # Try to find just numbers that might be CWE IDs
    number_pattern = r"\b(\d+)\b"
    matches = re.findall(number_pattern, text)
    if matches:
        # Take the first number found
        return f"CWE-{matches[0]}"

    return ""


@scorer(metrics=[accuracy(), stderr()])
def cti_bench_rcm_scorer() -> Callable:
    """Scorer for CTI-Bench RCM (CVE→CWE mapping) task."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract CWE ID from model response
        extracted_cwe = extract_cwe_id(state.output.completion)
        target_cwe = target.text.strip()

        # Normalize both to ensure consistent format
        if extracted_cwe and not extracted_cwe.startswith("CWE-"):
            extracted_cwe = f"CWE-{extracted_cwe}"
        if target_cwe and not target_cwe.startswith("CWE-"):
            target_cwe = f"CWE-{target_cwe}"

        # Check if extracted CWE matches target
        is_correct = extracted_cwe.upper() == target_cwe.upper()

        return Score(
            value=1.0 if is_correct else 0.0,
            answer=extracted_cwe,
            metadata={
                "extracted_cwe": extracted_cwe,
                "target_cwe": target_cwe,
                "raw_output": state.output.completion,
            },
        )

    return score


# VSP (CVSS severity prediction) Functions
def extract_cvss_score(text: str) -> float:
    """Extract CVSS score from model output."""
    if not text:
        return 0.0

    # Try to find decimal numbers (CVSS scores)
    decimal_pattern = r"(\d+\.\d+)"
    matches = re.findall(decimal_pattern, text)
    if matches:
        try:
            score = float(matches[0])
            # Clamp to valid CVSS range
            return max(0.0, min(10.0, score))
        except ValueError:
            pass

    # Try to find integers that might be CVSS scores
    integer_pattern = r"\b(\d+)\b"
    matches = re.findall(integer_pattern, text)
    if matches:
        try:
            score = float(matches[0])
            # Clamp to valid CVSS range
            return max(0.0, min(10.0, score))
        except ValueError:
            pass

    return 0.0


@scorer(metrics=[mean_absolute_deviation(), accuracy_within_threshold(), stderr()])
def cti_bench_vsp_scorer() -> Callable:
    """Scorer for CTI-Bench VSP (CVSS severity prediction) task."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract CVSS score from model response
        predicted_score = extract_cvss_score(state.output.completion)

        try:
            actual_score = float(target.text.strip())
        except ValueError:
            actual_score = 0.0

        # Calculate absolute deviation
        absolute_deviation = abs(predicted_score - actual_score)

        # Score is inversely related to deviation (lower deviation = higher score)
        # Use a score of 1.0 if deviation is 0, decreasing linearly
        score_value = max(0.0, 1.0 - (absolute_deviation / 10.0))

        return Score(
            value=score_value,
            answer=str(predicted_score),
            metadata={
                "predicted_score": predicted_score,
                "actual_score": actual_score,
                "absolute_deviation": absolute_deviation,
                "raw_output": state.output.completion,
            },
        )

    return score
