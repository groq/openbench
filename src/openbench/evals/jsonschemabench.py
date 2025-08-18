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
def jsonschemabench(subset: str | None = None, split: str = "all", num_shots: int = 0, strip_markdown: bool = True) -> Task:
    """JSONSchemaBench: JSON Schema generation benchmark.
    
    Evaluates the ability of language models to generate valid JSON
    that conforms to provided JSON schemas. Based on ~10K real-world
    schemas from GitHub, Kubernetes, APIs, and other sources.
    
    Following the paper methodology:
    - Zero-shot: "You need to generate a JSON object that matches the schema below."
    - Few-shot: Includes examples with "## Input Schema:" and "## Expected Output:" format
    
    Args:
        subset: Specific subset to evaluate (e.g., "Github_easy", "Kubernetes")
                or None for mixed benchmark
        split: Dataset split to use ("all", "test", "val", "train")
        num_shots: Number of few-shot examples to include (0 for zero-shot, paper used 2)
        strip_markdown: Whether to remove ```json``` markdown blocks from output (default True)
    
    Returns:
        Task configured for JSONSchemaBench evaluation
    """
    return Task(
        dataset=get_dataset(subset=subset, split=split, num_shots=num_shots),
        solver=[generate()],
        scorer=json_schema_scorer(strip_markdown=strip_markdown),
        name="jsonschemabench",
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for structured output
            max_tokens=4096,  # Allow space for complex JSON objects
        ),
    )