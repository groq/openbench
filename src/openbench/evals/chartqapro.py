"""ChartQAPro evaluation task."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task

from openbench.datasets.chartqapro import get_chartqapro_dataset
from openbench.scorers.chartqapro import chartqapro_scorer
from openbench.solvers.chartqapro import chartqapro_solver


@task
def chartqapro(prompt_strategy: Literal["direct", "cot", "pot"] = "direct") -> Task:
    """
    ChartQAPro: Comprehensive chart understanding benchmark.

    Tests multimodal reasoning across 5 question types with 1,950 samples from 157 diverse sources.

    Question Types:
        - Conversational: Multi-turn context-dependent questions (e.g., "What is this?", "How does this compare?")
        - Factoid: Single factual questions about chart data
        - Multi Choice: MCQ with options a, b, c, d
        - Fact Checking: True/false statements about chart content
        - Hypothetical: What-if scenarios based on chart trends

    Prompting Strategies:
        - "direct": Concise answer only (default, fastest)
        - "cot": Chain-of-thought reasoning (step-by-step with "The answer is X")
        - "pot": Program-of-thought (generates executable Python code)

    Scoring (per official evaluation):
        - MCQ/Fact Checking: Exact match (case-insensitive)
        - Years: Exact match when flagged in dataset
        - Numeric: 5% relative error tolerance (except years)
        - Text: ANLS (Average Normalized Levenshtein Similarity) with 0.5 threshold
        - List answers: Element-wise scoring averaged

    Dataset:
        - 1,950 test samples
        - 157 diverse sources (reports, dashboards, infographics)
        - 1-7 questions per chart
        - Optional context paragraphs
        - Embedded JPEG images

    Benchmark Stats:
        - Claude Sonnet 3.5: 55.81% (reported in paper)
        - GPT-4V: Performance varies by category
        - Challenging multi-turn reasoning required

    Args:
        prompt_strategy: Prompting approach ("direct", "cot", or "pot")

    Returns:
        Configured Task with dataset, solver, and scorer

    Example:
        # Direct prompting (default)
        bench eval chartqapro --model groq/llama-3.1-70b-vision

        # Chain-of-thought
        bench eval chartqapro --model groq/llama-3.1-70b-vision -T prompt_strategy=cot

        # Program-of-thought
        bench eval chartqapro --model groq/llama-3.1-70b-vision -T prompt_strategy=pot

        # Test with limited samples
        bench eval chartqapro --model groq/llama-3.1-70b-vision --limit 10

    Reference:
        Paper: https://arxiv.org/abs/2504.05506
        Dataset: https://huggingface.co/datasets/ahmed-masry/ChartQAPro
        Code: https://github.com/vis-nlp/ChartQAPro
    """
    dataset = get_chartqapro_dataset()

    return Task(
        dataset=dataset,
        solver=chartqapro_solver(prompt_strategy=prompt_strategy),
        scorer=chartqapro_scorer(),
        name=f"chartqapro_{prompt_strategy}",
    )
