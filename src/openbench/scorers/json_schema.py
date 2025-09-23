import json
from typing import Callable
from jsonschema import Draft202012Validator, ValidationError, FormatChecker
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    CORRECT,
    INCORRECT,
    accuracy,
    stderr,
)
from openbench.metrics.json_schema import (
    json_validity,
    schema_compliance,
    api_success_rate,
)


def _strip_markdown(text: str) -> str:
    """Strip markdown code blocks from text."""
    markdown_prefix = "```json"
    markdown_suffix = "```"
    return text.removeprefix(markdown_prefix).removesuffix(markdown_suffix)


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
