"""OCRBench v2 evaluation task."""

from __future__ import annotations

from inspect_ai import Task, task

from openbench.datasets.ocrbenchv2 import get_ocrbenchv2_dataset
from openbench.scorers.ocrbenchv2 import ocrbenchv2_scorer
from openbench.solvers.ocrbenchv2 import ocrbenchv2_solver


@task
def ocrbenchv2(
    categories: list[str] | None = None,
    language: str = "all",
    seed: int = 42,
) -> Task:
    """
    OCRBench v2: Visual text localization and reasoning benchmark.

    Comprehensive evaluation of vision-language models across 31 diverse OCR and document
    understanding scenarios including street scenes, receipts, formulas, diagrams, tables,
    charts, and document parsing.

    Categories:
        - English (21 tasks): VQA, text recognition, document parsing, tables, charts, math, etc.
        - Chinese (10 tasks): OCR, document parsing, formulas, handwriting, translation, etc.

    Args:
        categories: Filter specific categories (e.g., ["APP agent en", "table parsing en"])
        language: Filter by language - "en", "cn", or "all" (default: "all")
        seed: Random seed for shuffling dataset (default: 42, helps get diverse samples)

    Returns:
        Task with dataset, solver, and scorer

    Example:
        # Run full benchmark
        bench eval ocrbenchv2 --model openai/gpt-4o

        # Test with limited samples (gets diverse categories due to shuffling)
        bench eval ocrbenchv2 --model openai/gpt-4o --limit 15

        # English tasks only
        bench eval ocrbenchv2 --model openai/gpt-4o -T language=en

        # Chinese tasks only
        bench eval ocrbenchv2 --model openai/gpt-4o -T language=cn

        # Specific categories
        bench eval ocrbenchv2 --model openai/gpt-4o -T 'categories=["table parsing en","chart parsing en"]'

    Reference:
        Paper: https://arxiv.org/abs/2501.00321
        Dataset: https://huggingface.co/datasets/morpheushoc/OCRBenchv2
        Code: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2
    """
    dataset = get_ocrbenchv2_dataset(
        categories=categories, language=language, seed=seed
    )

    return Task(
        dataset=dataset,
        solver=ocrbenchv2_solver(),
        scorer=ocrbenchv2_scorer(),
        name=f"ocrbenchv2_{language}",
    )
