import json
import re
import ast
from collections import OrderedDict
from typing import Any

from inspect_ai.scorer import Score, scorer, accuracy, stderr
from inspect_ai.solver import TaskState
from openbench.metrics.clockbench import (
    compute_detailed_scores,
    convert_to_int_or_none,
    is_finite_number,
)

# mapping of expected fields by task
FIELDS_BY_TASK = {
    "time_fields": ["valid", "hours", "minutes", "seconds", "date", "month", "weekday"],
    "shift_fields": ["valid", "hours", "minutes", "seconds"],
    "angle_fields": ["valid", "hours", "minutes", "seconds"],
    "zone_fields": ["valid", "hours", "minutes", "seconds"],
}


def compare_gt_pred(ground_truth_obj, predicted_obj, required_fields):
    """Compare ground truth and predicted answer objects field by field."""

    ground_truth = normalize_answer_fields(ground_truth_obj, required_fields)
    predicted = normalize_answer_fields(predicted_obj, required_fields)

    comparison_details = OrderedDict()
    comparison_details["valid"] = (ground_truth.get("valid"), predicted.get("valid"))

    # validity comparison - must agree on valid/invalid
    if ground_truth.get("valid") is not predicted.get("valid"):
        return False, {**comparison_details, "reason": "validity_mismatch"}

    # if ground truth says invalid, other fields don't matter, return true
    if ground_truth.get("valid") is False:
        return True, comparison_details

    # if valid time, check all other fields
    all_fields_correct = True
    for field_name in required_fields:
        if field_name == "valid":
            continue

        field_matches = match_value(
            ground_truth.get(field_name), predicted.get(field_name)
        )
        comparison_details[field_name] = (
            ground_truth.get(field_name),
            predicted.get(field_name),
            field_matches,
        )
        all_fields_correct = all_fields_correct and field_matches

    return all_fields_correct, comparison_details


def parse_obj(value):
    """Parse potentially messy JSON from model responses."""
    if isinstance(value, dict):
        return value

    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json|javascript|js)?\s*|\s*```$", "", text, flags=re.I | re.S
        )

    json_match = re.search(r"\{.*\}", text, flags=re.S)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fix trailing commas
    text_fixed_commas = re.sub(r",(\s*[}\]])", r"\1", text)
    text_fixed_commas = re.sub(
        r"(?m)(?<=\{|,)\s*([A-Za-z_]\w*)\s*:", r'"\1":', text_fixed_commas
    )
    try:
        return json.loads(text_fixed_commas)
    except json.JSONDecodeError:
        pass

    # convert JS booleans to Python
    text_python_bools = re.sub(r"\btrue\b", "True", text_fixed_commas, flags=re.I)
    text_python_bools = re.sub(r"\bfalse\b", "False", text_python_bools, flags=re.I)
    text_python_bools = re.sub(r"\bnull\b", "None", text_python_bools, flags=re.I)
    return ast.literal_eval(text_python_bools)


def match_value(expected_value, actual_value):
    """Compare expected and actual values with flexible matching rules."""

    # strings: case-insensitive comparison
    if isinstance(expected_value, str):
        return (
            isinstance(actual_value, str)
            and expected_value.strip().casefold()
            == str(actual_value).strip().casefold()
        )

    # booleans and None: exact comparison
    if isinstance(expected_value, (bool, type(None))):
        return expected_value == actual_value

    # numeric comparison
    if is_finite_number(expected_value):
        actual_int = convert_to_int_or_none(actual_value)
        return actual_int is not None and actual_int == int(expected_value)

    # list comparison with inclusive range or choices
    if isinstance(expected_value, list) and expected_value:
        if len(expected_value) == 2 and all(
            is_finite_number(x) for x in expected_value
        ):
            # range comparison: [4, 5] means 4 <= actual <= 5
            actual_int = convert_to_int_or_none(actual_value)
            if actual_int is None:
                return False
            range_low, range_high = int(expected_value[0]), int(expected_value[1])
            return range_low <= actual_int <= range_high

        # multiple choice comparison: [4, 5, 6] means actual must be one of these
        valid_choices = {
            int(choice)
            for choice in expected_value
            if is_finite_number(choice)
            or (isinstance(choice, str) and re.fullmatch(r"-?\d+", choice))
        }
        actual_int = convert_to_int_or_none(actual_value)
        return actual_int is not None and actual_int in valid_choices

    # dictionary alternatives comparison
    if isinstance(expected_value, dict) and expected_value:
        choice_set = set()
        for dict_value in expected_value.values():
            if is_finite_number(dict_value):
                choice_set.add(int(dict_value))
            elif isinstance(dict_value, str) and re.fullmatch(
                r"-?\d+", dict_value.strip()
            ):
                choice_set.add(int(dict_value.strip()))
            elif (
                isinstance(dict_value, list)
                and len(dict_value) == 2
                and all(is_finite_number(x) for x in dict_value)
            ):
                range_low, range_high = int(dict_value[0]), int(dict_value[1])
                choice_set.update(range(range_low, range_high + 1))

        actual_int = convert_to_int_or_none(actual_value)
        return (
            (actual_int is not None and actual_int in choice_set)
            if choice_set
            else (expected_value == actual_value)
        )

    # fallback: exact comparison
    return expected_value == actual_value


def normalize_answer_fields(answer_dict, required_fields):
    """Extract and normalize required fields from answer dictionary."""
    normalized = {}
    for field_name in required_fields:
        normalized[field_name] = answer_dict.get(field_name, None)
    return normalized


@scorer(metrics=[accuracy(), stderr(), compute_detailed_scores()])
def clockbench_scorer():
    """
    Detailed clockbench scorer that stores full comparison results for sophisticated analysis.

    This scorer:
    1. Parses target fields from metadata and model response from solver output
    2. Uses original compare_gt_pred logic with detailed comparison tracking
    3. Stores full results structure for compute_detailed_scores analysis
    4. Returns sample-level accuracy with detailed metadata
    """

    async def score(state: TaskState, _target: Any) -> Score:
        try:
            # Parse target fields from metadata and model response from solver output
            model_responses = json.loads(state.output.completion)
            target_dict = state.metadata.get("target", {})

            question_types = ["time", "shift", "angle", "zone"]
            detailed_results = OrderedDict()
            per_task_scores = {}

            for question_type in question_types:
                # Extract expected fields based on task type
                task_fields = f"{question_type}_fields"
                required_fields = FIELDS_BY_TASK[task_fields]

                # Get ground truth from metadata and model response from solver output
                gt_data = target_dict.get(question_type, {})
                model_data = model_responses.get(question_type, {})

                # Parse model response - already parsed from JSON
                parsed_model_data = model_data

                # Use original detailed comparison logic
                is_correct, comparison_details = compare_gt_pred(
                    gt_data, parsed_model_data, required_fields
                )

                # Store detailed results in the format expected by compute_detailed_scores
                detailed_results[question_type] = {
                    "expected": gt_data,
                    "got": parsed_model_data,
                    "correct": is_correct,
                    "details": comparison_details,
                }

                # Store per-task score (1.0 for correct, 0.0 for incorrect)
                per_task_scores[question_type] = 1.0 if is_correct else 0.0

            # Overall accuracy for the sample
            sample_accuracy = sum(per_task_scores.values()) / len(question_types)

            # Store sample ID and detailed results
            sample_id = getattr(state, "sample_id", None) or f"sample_{id(state)}"

            return Score(
                value=sample_accuracy,
                metadata={
                    "sample_id": sample_id,
                    "detailed_results": detailed_results,
                    **per_task_scores,  # Include individual task scores in metadata
                },
            )

        except Exception as e:
            return Score(
                value=0.0,
                metadata={
                    "error": str(e),
                    "sample_id": getattr(state, "sample_id", "unknown"),
                },
            )

    return score
