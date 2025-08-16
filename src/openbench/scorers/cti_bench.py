"""CTI-Bench scorers for cybersecurity benchmarking tasks."""

import re
from typing import Callable, List, Set
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target, Metric, metric
from inspect_ai.solver import TaskState
from inspect_ai.scorer._metric import SampleScore, Value


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


@metric
def technique_precision() -> Metric:
    """Calculate precision for technique extraction."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"technique_precision": 0.0}

        total_precision = 0.0
        valid_samples = 0

        for score in scores:
            metadata = score.score.metadata or {}
            predicted = set(metadata.get("predicted_techniques", []))
            ground_truth = set(metadata.get("ground_truth_techniques", []))

            if predicted:
                precision = len(predicted & ground_truth) / len(predicted)
                total_precision += precision
                valid_samples += 1
            elif not ground_truth:
                # If no predictions and no ground truth, count as perfect precision
                total_precision += 1.0
                valid_samples += 1

        if valid_samples == 0:
            return {"technique_precision": 0.0}

        avg_precision = total_precision / valid_samples
        return {"technique_precision": round(avg_precision, 4)}

    return metric_fn


@metric
def technique_recall() -> Metric:
    """Calculate recall for technique extraction."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"technique_recall": 0.0}

        total_recall = 0.0
        valid_samples = 0

        for score in scores:
            metadata = score.score.metadata or {}
            predicted = set(metadata.get("predicted_techniques", []))
            ground_truth = set(metadata.get("ground_truth_techniques", []))

            if ground_truth:
                recall = len(predicted & ground_truth) / len(ground_truth)
                total_recall += recall
                valid_samples += 1
            elif not predicted:
                # If no ground truth and no predictions, count as perfect recall
                total_recall += 1.0
                valid_samples += 1

        if valid_samples == 0:
            return {"technique_recall": 0.0}

        avg_recall = total_recall / valid_samples
        return {"technique_recall": round(avg_recall, 4)}

    return metric_fn


@metric
def technique_f1() -> Metric:
    """Calculate F1 score for technique extraction."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"technique_f1": 0.0}

        # Calculate individual precision and recall for each sample
        total_f1 = 0.0
        valid_samples = 0

        for score in scores:
            metadata = score.score.metadata or {}
            predicted = set(metadata.get("predicted_techniques", []))
            ground_truth = set(metadata.get("ground_truth_techniques", []))

            if not predicted and not ground_truth:
                # Perfect match when both are empty
                f1 = 1.0
            elif not predicted or not ground_truth:
                # One is empty, the other is not - F1 is 0
                f1 = 0.0
            else:
                # Both have values, calculate F1
                tp = len(predicted & ground_truth)
                precision = tp / len(predicted) if predicted else 0.0
                recall = tp / len(ground_truth) if ground_truth else 0.0

                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

            total_f1 += f1
            valid_samples += 1

        if valid_samples == 0:
            return {"technique_f1": 0.0}

        avg_f1 = total_f1 / valid_samples
        return {"technique_f1": round(avg_f1, 4)}

    return metric_fn


@metric
def exact_match_accuracy() -> Metric:
    """Calculate exact match accuracy for technique extraction."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"exact_match_accuracy": 0.0}

        exact_matches = 0
        for score in scores:
            metadata = score.score.metadata or {}
            predicted = set(metadata.get("predicted_techniques", []))
            ground_truth = set(metadata.get("ground_truth_techniques", []))

            if predicted == ground_truth:
                exact_matches += 1

        accuracy = exact_matches / len(scores)
        return {"exact_match_accuracy": round(accuracy, 4)}

    return metric_fn


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


@metric
def mean_absolute_deviation() -> Metric:
    """Calculate Mean Absolute Deviation for CVSS score predictions."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"mean_absolute_deviation": 0.0}

        deviations = []
        for score in scores:
            if hasattr(score, "metadata") and score.metadata:
                predicted = score.metadata.get("predicted_score", 0.0)
                actual = score.metadata.get("actual_score", 0.0)
                deviation = abs(predicted - actual)
                deviations.append(deviation)

        if not deviations:
            return {"mean_absolute_deviation": 0.0}

        mad = sum(deviations) / len(deviations)
        return {"mean_absolute_deviation": round(mad, 4)}

    return metric_fn


@metric
def accuracy_within_threshold() -> Metric:
    """Calculate accuracy within 1.0 CVSS point threshold."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return {"accuracy_within_1_point": 0.0}

        correct = 0
        for score in scores:
            if hasattr(score, "metadata") and score.metadata:
                predicted = score.metadata.get("predicted_score", 0.0)
                actual = score.metadata.get("actual_score", 0.0)
                if abs(predicted - actual) <= 1.0:
                    correct += 1

        accuracy = correct / len(scores)
        return {"accuracy_within_1_point": round(accuracy, 4)}

    return metric_fn


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
