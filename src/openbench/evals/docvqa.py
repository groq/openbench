"""DocVQA (Document Visual Question Answering) evaluation task.

DocVQA evaluates a model's ability to answer questions about document images
including forms, reports, tables, diagrams, and other real-world documents.

Reference: https://arxiv.org/abs/2007.00398
Homepage: https://www.docvqa.org/
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import generate

from openbench.datasets.docvqa import get_docvqa_dataset
from openbench.metrics.docvqa import anls_by_question_type
from openbench.scorers.docvqa import docvqa_anls


@task
def docvqa(
    split: str = "validation",
) -> Task:
    """DocVQA: Document Visual Question Answering benchmark.

    Evaluates models on answering questions about document images using the
    ANLS (Average Normalized Levenshtein Similarity) metric. The benchmark
    includes diverse document types: forms, reports, tables, diagrams, etc.

    The evaluation uses:
    - ANLS scoring with threshold 0.5 (handles OCR errors gracefully)
    - Case-insensitive but space-sensitive matching
    - Maximum similarity across multiple acceptable answers
    - Breakdown by question type (figure/diagram, table, form, etc.)

    Human performance: 94.36% ANLS (from original paper)

    Args:
        split: Dataset split - "validation" (5,349 samples) or "test" (5,188 samples)
               Note: test split has no answers, for leaderboard submission only

    Returns:
        Task configured for DocVQA evaluation
    """
    # Load dataset
    dataset = get_docvqa_dataset(split=split)

    return Task(
        dataset=dataset,
        solver=[
            generate(),
        ],
        scorer=docvqa_anls(),
        metrics=[
            accuracy(),  # Overall ANLS score (average across all samples)
            stderr(),  # Standard error
            anls_by_question_type(),  # ANLS breakdown by document type
        ],
        config=GenerateConfig(
            max_tokens=512,  # DocVQA answers are typically short (1-20 tokens)
        ),
    )
