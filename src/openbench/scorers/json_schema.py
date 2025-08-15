import json
from typing import Callable
from jsonschema import Draft202012Validator, ValidationError, FormatChecker
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
    CORRECT,
    INCORRECT,
    accuracy,
    stderr,
)


@metric
def json_validity() -> Metric:
    """Calculates the percentage of outputs that are valid JSON."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        json_valid_count = sum(
            1
            for score in scores
            if score.score.metadata and score.score.metadata.get("json_valid", False)
        )
        return json_valid_count / len(scores)

    return metric_calculator


@metric
def schema_compliance() -> Metric:
    """Calculates the percentage of valid JSON outputs that conform to schema."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        valid_json_scores = [
            score
            for score in scores
            if score.score.metadata and score.score.metadata.get("json_valid", False)
        ]

        if not valid_json_scores:
            return 0.0

        schema_compliant_count = sum(
            1
            for score in valid_json_scores
            if score.score.metadata
            and score.score.metadata.get("schema_compliant", False)
        )
        return schema_compliant_count / len(valid_json_scores)

    return metric_calculator


@metric
def overall_success() -> Metric:
    """Calculates the percentage of outputs that are both valid JSON and schema compliant."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        success_count = sum(
            1
            for score in scores
            if (
                score.score.metadata
                and score.score.metadata.get("json_valid", False)
                and score.score.metadata.get("schema_compliant", False)
            )
        )
        return success_count / len(scores)

    return metric_calculator


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        json_validity(),
        schema_compliance(),
        overall_success(),
    ]
)
def json_schema_scorer() -> Callable:
    """
    Scorer that validates JSON output against a provided schema.

    Follows JSONSchemaBench methodology:
    - Uses Draft2020-12 validator with format checking
    - Returns separate metrics for JSON validity and schema compliance
    - Single attempt per schema (no retries)

    Expects schema in state.metadata["schema"]
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Extract schema from sample metadata
        if not state.metadata or "schema" not in state.metadata:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                metadata={
                    "json_valid": False,
                    "schema_compliant": False,
                    "error": "no_schema",
                },
            )

        schema_data = state.metadata["schema"]
        # Handle both string (from dataset) and dict (from tests) formats
        schema = json.loads(schema_data) if isinstance(schema_data, str) else schema_data
        raw_output = state.output.completion
        processed_output = raw_output.strip() if raw_output else ""

        # Step 1: Check if output is valid JSON
        try:
            json_data = json.loads(processed_output)
            json_valid = True
        except (json.JSONDecodeError, ValueError) as e:
            return Score(
                value=INCORRECT,
                answer=raw_output,
                metadata={
                    "json_valid": False,
                    "schema_compliant": False,
                    "error": f"json_decode_error: {str(e)}",
                },
            )

        # Step 2: Validate against schema using JSONSchemaBench methodology
        try:
            # Use Draft2020-12 with format checking (as per JSB paper)
            validator = Draft202012Validator(schema, format_checker=FormatChecker())
            validator.validate(json_data)
            schema_compliant = True
            error_msg = None
        except ValidationError as e:
            schema_compliant = False
            error_msg = f"schema_validation_error: {e.message}"
        except Exception as e:
            schema_compliant = False
            error_msg = f"validation_error: {str(e)}"

        # Return score with detailed metadata
        success = json_valid and schema_compliant
        return Score(
            value=CORRECT if success else INCORRECT,
            answer=raw_output,  # Always store raw output for debugging
            metadata={
                "json_valid": json_valid,
                "schema_compliant": schema_compliant,
                "error": error_msg,
            },
        )

    return score
