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

from openbench.datasets.jsonschemabench import get_dataset, Compatibility
from openbench.scorers.json_schema import json_schema_scorer


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


@task
def jsonschemabench(
    subset: str | None = None,
    split: str = "all",
    num_shots: int = 0,
    strip_markdown: bool = True,
    use_response_schema: bool = False,
    strict: bool = False,
    adapt_schema: bool = True,
    compatibility: Compatibility = Compatibility.DEFAULT,
) -> Task:
    """JSONSchemaBench: A Rigorous Benchmark of Structured Outputs
    for Language Models.

    Evaluates the ability of language models to generate valid JSON
    that conforms to provided JSON schemas. Based on ~10K real-world
    schemas from GitHub, Kubernetes, APIs, and other sources.

    Uses structured output when supported by the provider for API-level
    schema validation, otherwise falls back to text generation with post-hoc validation.

    See https://doi.org/10.48550/arXiv.2501.10868.

    Args:
        subset: Specific subset to evaluate (e.g., "Github_easy", "Kubernetes")
                or None for mixed benchmark
        split: Dataset split to use ("all", "test", "val", "train")
        num_shots: Number of few-shot examples to include (0 for zero-shot, paper used 2)
        strip_markdown: Whether to remove ```json``` markdown blocks from output (default True)
        use_response_schema: Whether to use ResponseSchema validation when supported (default False)
        strict: Whether to use strict mode for structured output (default False)
        adapt_schema: Whether to adapt schemas for better provider compatibility (default True)
        compatibility: Filter to records compatible with specific APIs (default: no filtering)

    Returns:
        Task configured for JSONSchemaBench evaluation
    """
    return Task(
        dataset=get_dataset(
            subset=subset,
            split=split,
            num_shots=num_shots,
            adapt_schema=adapt_schema,
            compatibility=compatibility,
        ),
        solver=[
            response_schema_solver(
                use_response_schema=use_response_schema, strict=strict
            )
        ],
        scorer=json_schema_scorer(strip_markdown=strip_markdown),
        name="jsonschemabench",
        config=GenerateConfig(
            temperature=0.0,  # Following paper methodology (greedy decoding)
            timeout=40,  # 40-second timeout as per original paper
        ),
    )
