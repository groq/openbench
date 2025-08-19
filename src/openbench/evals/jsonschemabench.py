"""JSONSchemaBench: JSON Schema generation benchmark evaluation.

Based on: JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models
EPFL DLAB, 2025
https://arxiv.org/html/2501.10868

Dataset: https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench
"""

import json
from jsonschema import Draft202012Validator
from inspect_ai import Task, task
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.model import GenerateConfig, ResponseSchema, ModelOutput

from openbench.datasets.jsonschemabench import get_dataset
from openbench.scorers.json_schema import json_schema_scorer


def add_root_type_if_missing(schema: dict) -> None:
    """Add type: object if missing from schema root."""
    if "type" not in schema:
        schema["type"] = "object"


def recursively_set_additional_properties_false(schema: dict) -> None:
    """Recursively add additionalProperties: false to objects with properties."""
    if not isinstance(schema, dict):
        return
    # Set additionalProperties to false if it's missing or true, and object has properties
    if schema.get("properties") and (
        "additionalProperties" not in schema or schema.get("additionalProperties", True)
    ):
        schema["additionalProperties"] = False
    # Recurse into properties
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    # Recurse into array items
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


def set_all_properties_required(schema: dict) -> dict:
    """Recursively make all properties required in objects."""
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema


def adapt_schema_for_openai(schema_dict: dict) -> dict:
    """Adapt schema using JSONSchemaBench-style modifications for OpenAI compatibility."""
    import copy
    adapted_schema = copy.deepcopy(schema_dict)
    add_root_type_if_missing(adapted_schema)
    recursively_set_additional_properties_false(adapted_schema)
    adapted_schema = set_all_properties_required(adapted_schema)
    return adapted_schema


@solver
def structured_output_solver(use_structured_output: bool = True, strict: bool = False, adapt_schema: bool = False):
    """Apply per-sample structured output for supported providers (OpenAI, Google, Mistral)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.metadata or "schema" not in state.metadata:
            return await generate(state)

        # Skip structured output if disabled
        if not use_structured_output:
            return await generate(state)

        try:
            schema_str = state.metadata["schema"]
            schema_dict = json.loads(schema_str)
            
            # Assert that it's a valid JSON Schema
            Draft202012Validator.check_schema(schema_dict)
            
            # Apply schema adaptation if enabled and using structured output
            if adapt_schema and use_structured_output:
                schema_dict = adapt_schema_for_openai(schema_dict)

            return await generate(
                state,
                response_schema=ResponseSchema(
                    name="json_schema_output", json_schema=schema_dict, strict=strict
                ),
            )

        except Exception as e:
            # Schema validation failed - mark as API error instead of falling back
            error_msg = f"schema_validation_error (strict={strict}): {str(e)}"
            
            # CSV format: dataset, split, strict, id, error, schema
            dataset_info = getattr(state, 'sample_id', 'unknown')
            subset = state.metadata.get('subset', 'unknown') if state.metadata else 'unknown'
            schema_preview = schema_str.replace('"', '""').replace('\n', ' ')[:200] + "..." if len(schema_str) > 200 else schema_str.replace('"', '""').replace('\n', ' ')
            csv_error = str(e).replace('"', '""').replace('\n', ' ')  # Escape CSV
            csv_log = f"jsonschemabench,all,{strict},{dataset_info},{csv_error},\"{schema_preview}\""
            print(csv_log)
            
            state.output = ModelOutput.from_content(
                model="", content="", error=error_msg
            )
            return state

    return solve


@task
def jsonschemabench(
    subset: str | None = None,
    split: str = "all",
    num_shots: int = 0,
    strip_markdown: bool = True,
    use_structured_output: bool = True,
    strict: bool = False,
    adapt_schema: bool = False,
) -> Task:
    """JSONSchemaBench: A Rigorous Benchmark of Structured Outputs
    for Language Models.

    Evaluates the ability of language models to generate valid JSON
    that conforms to provided JSON schemas. Based on ~10K real-world
    schemas from GitHub, Kubernetes, APIs, and other sources.

    Uses structured output when supported by the provider for API-level
    schema validation, otherwise falls back to text generation withpost-hoc validation.

    See https://doi.org/10.48550/arXiv.2501.10868.

    Args:
        subset: Specific subset to evaluate (e.g., "Github_easy", "Kubernetes")
                or None for mixed benchmark
        split: Dataset split to use ("all", "test", "val", "train")
        num_shots: Number of few-shot examples to include (0 for zero-shot, paper used 2)
        strip_markdown: Whether to remove ```json``` markdown blocks from output (default True)
        use_structured_output: Whether to use structured output when supported (default True)
        strict: Whether to use strict mode for structured output (default True)
        adapt_schema: Whether to adapt schemas for better provider compatibility (default False)

    Returns:
        Task configured for JSONSchemaBench evaluation
    """
    return Task(
        dataset=get_dataset(subset=subset, split=split, num_shots=num_shots),
        solver=[structured_output_solver(use_structured_output=use_structured_output, strict=strict, adapt_schema=adapt_schema)],
        scorer=json_schema_scorer(strip_markdown=strip_markdown),
        name="jsonschemabench",
        config=GenerateConfig(
            temperature=0.0,  # Following paper methodology (greedy decoding)
            timeout=40,  # 40-second timeout as per original paper
        ),
    )
