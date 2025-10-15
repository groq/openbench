"""JSONSchemaBench solver."""

from __future__ import annotations

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import ResponseSchema, ModelOutput
import json
from jsonschema import Draft202012Validator


@solver
def response_schema_solver(use_response_schema: bool = False, strict: bool = False):
    """Apply per-sample ResponseSchema for supported providers (OpenAI, Google, Mistral)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.metadata or "schema" not in state.metadata:
            return await generate(state)

        # Skip ResponseSchema if disabled
        if not use_response_schema:
            return await generate(state)

        try:
            schema_str = state.metadata["schema"]
            schema_dict = json.loads(schema_str)

            # Assert that it's a valid JSON Schema
            Draft202012Validator.check_schema(schema_dict)

            return await generate(
                state,
                response_schema=ResponseSchema(
                    name="json_schema_output", json_schema=schema_dict, strict=strict
                ),
            )

        except Exception as e:
            # Schema validation failed - mark as API error instead of falling back
            error_msg = f"schema_validation_error (strict={strict}): {str(e)}"

            state.output = ModelOutput.from_content(
                model="", content="", error=error_msg
            )
            return state

    return solve
