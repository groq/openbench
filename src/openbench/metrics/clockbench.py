"""ClockBench metrics for evaluation scoring."""

import math
from collections import OrderedDict
from statistics import median
from typing import List

from inspect_ai.scorer import metric, Metric, Value
from inspect_ai.scorer._metric import SampleScore


# Helper functions for detailed analysis
def calculate_percentage(numerator, denominator):
    """Calculate percentage, handling division by zero."""
    return None if denominator == 0 else round(numerator / denominator, 4)


def calculate_fraction(numerator, denominator):
    """Calculate fraction, handling division by zero."""
    return None if denominator == 0 else round(numerator / denominator, 4)


def calculate_range_midpoint(range_low, range_high):
    """Get midpoint for ranges."""
    return int(round((int(range_low) + int(range_high)) / 2.0))


def is_finite_number(value):
    """Check if value is a finite number (not boolean, not NaN, not infinity)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def convert_to_int_or_none(value):
    """Convert value to integer if possible, otherwise return None."""
    if is_finite_number(value):
        return int(value)
    if isinstance(value, str):
        value_stripped = value.strip()
        if value_stripped.isdigit() or (
            value_stripped.startswith("-") and value_stripped[1:].isdigit()
        ):
            return int(value_stripped)
    return None


def convert_expected_value_to_scalar(expected_value):
    """Convert expected value to scalar, handling false/none as zero and ranges as midpoint."""
    if expected_value is False or expected_value is None:
        return 0

    expected_int = convert_to_int_or_none(expected_value)
    if expected_int is not None:
        return expected_int

    # handle range values like [4, 5] by taking midpoint
    if (
        isinstance(expected_value, list)
        and len(expected_value) == 2
        and convert_to_int_or_none(expected_value[0]) is not None
        and convert_to_int_or_none(expected_value[1]) is not None
    ):
        return calculate_range_midpoint(
            convert_to_int_or_none(expected_value[0]),
            convert_to_int_or_none(expected_value[1]),
        )
    return None


def convert_predicted_value_to_scalar(predicted_value):
    """Convert model prediction to scalar."""
    if predicted_value is False or predicted_value is None:
        return 0
    return convert_to_int_or_none(predicted_value)


def determine_clock_period_hours(sample_key, ground_truth_hours):
    """Determine if clock uses 24h or 12h format for wrap-around calculations."""
    hours_int = convert_to_int_or_none(ground_truth_hours)
    if hours_int is not None and hours_int >= 13:
        return 24

    key_lower = str(sample_key).lower()
    if ("24" in key_lower) and ("hour" in key_lower):
        return 24
    return 12


def convert_time_to_seconds(hours, minutes, seconds, period_hours):
    """Convert time components to total seconds within period."""
    hours_normalized = (int(hours) if hours is not None else 0) % period_hours
    minutes_normalized = int(minutes) if minutes is not None else 0
    seconds_normalized = int(seconds) if seconds is not None else 0

    # map onto [0, period_hours*3600) for circular time comparison
    return (hours_normalized * 3600 + minutes_normalized * 60 + seconds_normalized) % (
        period_hours * 3600
    )


def convert_sec_to_hours_min(total_sec):
    """Convert seconds to hours and minutes format."""
    if total_sec is None:
        return None
    total_sec_int = int(round(total_sec))
    hours = total_sec_int // 3600
    min_val = (total_sec_int % 3600) // 60
    return {"hours": hours, "minutes": min_val}


@metric
def compute_detailed_scores() -> Metric:
    """
    Computes clockbench metrics using the original scoring logic.

    Args:
        scores: list of SampleScore

    Returns:
        dict of clockbench metrics (including per-task accuracy, validity breakdown, time delta)
    """

    def metric_calculator(scores: List[SampleScore]) -> Value:
        # reconstruct all_results structure from sample metadata
        all_results: OrderedDict = OrderedDict()

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            sample_id = metadata.get("sample_id", f"sample_{len(all_results)}")

            if "detailed_results" in metadata:
                all_results[sample_id] = metadata["detailed_results"]

        if not all_results:
            return {}

        question_types = ["time", "shift", "angle", "zone"]
        sample_ids = list(all_results.keys())

        # basic accuracy breakdown
        totals = {q_type: {"correct": 0, "total": 0} for q_type in question_types}

        for sample_id in sample_ids:
            for q_type in question_types:
                if q_type in all_results[sample_id]:
                    totals[q_type]["total"] += 1
                    if all_results[sample_id][q_type]["correct"]:
                        totals[q_type]["correct"] += 1

        # validity breakdown (using time as base task)
        base_task = "time"

        valid_total = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["expected"].get("valid") is True
        )
        invalid_total = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["expected"].get("valid") is False
        )
        total_correct_base = sum(
            1
            for sample_id in sample_ids
            if all_results[sample_id][base_task]["correct"]
        )
        valid_correct = sum(
            1
            for sample_id in sample_ids
            if (
                all_results[sample_id][base_task]["expected"].get("valid") is True
                and all_results[sample_id][base_task]["correct"]
            )
        )
        invalid_correct = sum(
            1
            for sample_id in sample_ids
            if (
                all_results[sample_id][base_task]["expected"].get("valid") is False
                and all_results[sample_id][base_task]["correct"]
            )
        )

        validity_breakdown: OrderedDict = OrderedDict(
            [
                ("task", base_task),
                ("total_items", len(sample_ids)),
                ("total_correct", total_correct_base),
                (
                    "valid",
                    OrderedDict(
                        [
                            ("correct", valid_correct),
                            ("total", valid_total),
                            (
                                "accuracy",
                                calculate_percentage(valid_correct, valid_total),
                            ),
                        ]
                    ),
                ),
                (
                    "invalid",
                    OrderedDict(
                        [
                            ("correct", invalid_correct),
                            ("total", invalid_total),
                            (
                                "accuracy",
                                calculate_percentage(invalid_correct, invalid_total),
                            ),
                        ]
                    ),
                ),
            ]
        )

        # follow-up questions breakdown
        followup_types = ["shift", "angle", "zone"]

        valid_time_correct_ids = [
            sample_id
            for sample_id in sample_ids
            if all_results[sample_id]["time"]["correct"]
            and all_results[sample_id]["time"]["expected"].get("valid") is True
        ]
        valid_time_correct_count = len(valid_time_correct_ids)

        cond_accuracy: OrderedDict = OrderedDict()
        cond_accuracy["denominator_valid_time_correct"] = valid_time_correct_count

        for followup_type in followup_types:
            correct_count = sum(
                1
                for sample_id in valid_time_correct_ids
                if all_results[sample_id][followup_type]["correct"]
            )
            cond_accuracy[f"{followup_type}_given_valid_time_correct"] = {
                "numerator": correct_count,
                "denominator": valid_time_correct_count,
                "accuracy": calculate_fraction(correct_count, valid_time_correct_count),
            }

        # time delta breakdown
        circular_deltas = []
        excluded_alternatives = 0
        skipped_incomplete = 0

        for sample_id in sample_ids:
            gt_time = all_results[sample_id]["time"]["expected"]
            pred_time = all_results[sample_id]["time"]["got"]

            # keep only valid times
            if gt_time.get("valid") is not True:
                continue

            # exclude alternatives
            if any(
                isinstance(gt_time.get(field), dict)
                for field in ("hours", "minutes", "seconds")
            ):
                excluded_alternatives += 1
                continue

            # skip if already correct
            if all_results[sample_id]["time"]["correct"]:
                continue

            expected_h, expected_m, expected_s = (
                convert_expected_value_to_scalar(gt_time.get("hours")),
                convert_expected_value_to_scalar(gt_time.get("minutes")),
                convert_expected_value_to_scalar(gt_time.get("seconds")),
            )
            pred_h, pred_m, pred_s = (
                convert_predicted_value_to_scalar(pred_time.get("hours")),
                convert_predicted_value_to_scalar(pred_time.get("minutes")),
                convert_predicted_value_to_scalar(pred_time.get("seconds")),
            )

            if None in (expected_h, expected_m, expected_s, pred_h, pred_m, pred_s):
                skipped_incomplete += 1
                continue

            period_hours = determine_clock_period_hours(sample_id, gt_time.get("hours"))
            expected_total_sec = convert_time_to_seconds(
                expected_h, expected_m, expected_s, period_hours
            )
            pred_total_sec = convert_time_to_seconds(
                pred_h, pred_m, pred_s, period_hours
            )

            period_total_sec = period_hours * 3600
            raw_diff = abs(pred_total_sec - expected_total_sec)
            circular_deltas.append(min(raw_diff, period_total_sec - raw_diff))

        avg_delta_sec = (
            round(sum(circular_deltas) / len(circular_deltas), 2)
            if circular_deltas
            else None
        )
        median_delta_sec = (
            round(median(circular_deltas), 2) if circular_deltas else None
        )
        avg_delta_hm = convert_sec_to_hours_min(avg_delta_sec)
        median_delta_hm = convert_sec_to_hours_min(median_delta_sec)

        # invalid predictions breakdown
        predicted_invalid_count = 0
        for sample_id in sample_ids:
            predicted_time = all_results[sample_id]["time"]["got"]
            predicted_validity = (
                predicted_time.get("valid")
                if isinstance(predicted_time, dict)
                else None
            )
            if predicted_validity is False:
                predicted_invalid_count += 1

        predicted_invalid_percentage = (
            round(100 * predicted_invalid_count / len(sample_ids), 2)
            if sample_ids
            else 0.0
        )

        # final scores (following orig structure)
        per_task_accuracy = {
            f"{q_type}_accuracy": round(
                totals[q_type]["correct"] / max(1, totals[q_type]["total"]), 4
            )
            for q_type in question_types
        }

        result_scores = OrderedDict([("per_task_accuracy_abs", per_task_accuracy)])
        result_scores["time_validity_breakdown"] = validity_breakdown
        result_scores["predicted_invalid"] = {
            "count": predicted_invalid_count,
            "percent_of_all_items": predicted_invalid_percentage,
        }
        result_scores["conditional_accuracy_given_valid_answer_time_correct"] = (
            cond_accuracy
        )
        result_scores["time_delta_seconds_on_incorrect_valid_circular"] = OrderedDict(
            [
                ("count_items", len(circular_deltas)),
                ("average_delta_seconds", float(avg_delta_sec or 0.0)),
                ("median_delta_seconds", float(median_delta_sec or 0.0)),
                ("average_delta_hm", avg_delta_hm),
                ("median_delta_hm", median_delta_hm),
                ("excluded_due_to_alternatives", excluded_alternatives),
                ("skipped_incomplete_after_normalization", skipped_incomplete),
            ]
        )

        # extract values for metrics output
        return {
            "time_reading_accuracy": per_task_accuracy.get("time_accuracy", 0.0),
            "shift_accuracy": per_task_accuracy.get("shift_accuracy", 0.0),
            "angle_accuracy": per_task_accuracy.get("angle_accuracy", 0.0),
            "zone_accuracy": per_task_accuracy.get("zone_accuracy", 0.0),
            "predicted_invalid_time_percent": float(predicted_invalid_percentage),
            "average_time_error_seconds": float(avg_delta_sec or 0.0),
            "median_time_error_seconds": float(median_delta_sec or 0.0),
            # validity breakdown metrics (readable vs broken clocks)
            "readable_clocks_accuracy": validity_breakdown["valid"]["accuracy"] or 0.0,
            "broken_clocks_accuracy": validity_breakdown["invalid"]["accuracy"] or 0.0,
            # conditional accuracy (follow-up performance when initial time reading was correct)
            "correct_valid_time": cond_accuracy["denominator_valid_time_correct"],
            "conditional_shift_accuracy": cond_accuracy[
                "shift_given_valid_time_correct"
            ]["accuracy"]
            or 0.0,
            "conditional_angle_accuracy": cond_accuracy[
                "angle_given_valid_time_correct"
            ]["accuracy"]
            or 0.0,
            "conditional_zone_accuracy": cond_accuracy["zone_given_valid_time_correct"][
                "accuracy"
            ]
            or 0.0,
            # time error analysis details
            "predicted_incorrect_time": len(circular_deltas),
            "excluded_multiple_answers": excluded_alternatives,
            "skipped_incomplete_data": skipped_incomplete,
        }

    return metric_calculator
