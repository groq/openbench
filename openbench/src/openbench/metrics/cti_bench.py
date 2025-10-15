"""CTI-Bench metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import metric, Metric
from inspect_ai.scorer._metric import SampleScore, Value


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
