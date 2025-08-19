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


def _strip_markdown(text: str) -> str:
    """Strip markdown code blocks from text."""
    markdown_prefix = "```json"
    markdown_suffix = "```"
    return text.removeprefix(markdown_prefix).removesuffix(markdown_suffix)


@metric
def json_validity() -> Metric:
    """Calculates the percentage of successful API calls that produced valid JSON (empirical coverage)."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        # Get samples that had successful API calls (no API errors)
        successful_api_scores = [
            score
            for score in scores
            if score.score.metadata and not score.score.metadata.get("api_error", False)
        ]

        if not successful_api_scores:
            return 0.0

        json_valid_count = sum(
            1
            for score in successful_api_scores
            if score.score.metadata and score.score.metadata.get("json_valid", False)
        )
        return json_valid_count / len(successful_api_scores)

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
def api_success_rate() -> Metric:
    """Calculates the percentage of samples that didn't have API errors."""

    # TODO: Change this to only check for structured output related errors
    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        api_success_count = sum(
            1
            for score in scores
            if score.score.metadata and not score.score.metadata.get("api_error", False)
        )
        return api_success_count / len(scores)

    return metric_calculator


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        api_success_rate(),
        json_validity(),
        schema_compliance(),
    ]
)
def json_schema_scorer(strip_markdown: bool = True) -> Callable:
    """
    Scorer that validates JSON output against a provided schema.

    Follows JSONSchemaBench methodology:
    - Uses Draft2020-12 validator with format checking
    - Returns separate metrics for JSON validity and schema compliance
    - Optionally strips markdown code blocks from output

    Args:
        strip_markdown: Whether to remove ```json``` markdown blocks from output (default True)

    Expects schema in state.metadata["schema"]
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Check for API errors first (matches original paper's "declared coverage")
        if state.output.error:
            return Score(
                value=INCORRECT,
                answer=state.output.completion or "",
                metadata={
                    "json_valid": False,
                    "schema_compliant": False,
                    "api_error": True,
                    "error": f"api_error: {state.output.error}",
                },
            )

        # Extract schema from sample metadata
        if not state.metadata or "schema" not in state.metadata:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                metadata={
                    "json_valid": False,
                    "schema_compliant": False,
                    "api_error": False,
                    "error": "no_schema",
                },
            )

        schema_data = state.metadata["schema"]
        # Handle both string (from dataset) and dict (from tests) formats
        schema = (
            json.loads(schema_data) if isinstance(schema_data, str) else schema_data
        )
        raw_output = state.output.completion
        processed_output = raw_output.strip()
        processed_output = (
            _strip_markdown(processed_output) if strip_markdown else processed_output
        )

        # Check if output is valid JSON
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
                    "api_error": False,
                    "error": f"json_decode_error: {str(e)}",
                },
            )

        # Validate against schema using JSONSchemaBench methodology
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
                "api_error": False,
                "error": error_msg,
            },
        )

    return score
