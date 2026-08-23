"""JSONSchemaBench solver."""

from __future__ import annotations

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import ModelOutput
import json
from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

from openbench.model.response_schema import RawResponseSchema


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

            # Validate against the declared dialect, defaulting to the
            # benchmark's 2020-12 dialect when $schema is absent.
            validator_for(schema_dict, default=Draft202012Validator).check_schema(
                schema_dict
            )

            return await generate(
                state,
                response_schema=RawResponseSchema(
                    name="json_schema_output",
                    json_schema=schema_dict,
                    raw_json_schema=schema_dict,
                    strict=strict,
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
