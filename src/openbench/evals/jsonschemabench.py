"""JSONSchemaBench: JSON Schema generation benchmark evaluation.

Based on: JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models
EPFL DLAB, 2025
https://arxiv.org/html/2501.10868

Dataset: https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig

from openbench.datasets.jsonschemabench import get_dataset
from openbench.scorers.json_schema import json_schema_scorer


@task
def jsonschemabench(subset: str | None = None, split: str = "test") -> Task:
    """JSONSchemaBench: JSON Schema generation benchmark.
    
    Evaluates the ability of language models to generate valid JSON
    that conforms to provided JSON schemas. Based on ~10K real-world
    schemas from GitHub, Kubernetes, APIs, and other sources.
    
    Uses the prompt: "Generate a valid JSON object that matches the schema below."
    
    Args:
        subset: Specific subset to evaluate (e.g., "Github_easy", "Kubernetes")
               or None for mixed benchmark
        split: Dataset split to use ("test", "val", "train")
    
    Returns:
        Task configured for JSONSchemaBench evaluation
    """
    return Task(
        dataset=get_dataset(subset=subset, split=split),
        solver=[generate()],
        scorer=json_schema_scorer(),
        name="jsonschemabench",
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for structured output
            max_tokens=4096,  # Allow space for complex JSON objects
        ),
    )