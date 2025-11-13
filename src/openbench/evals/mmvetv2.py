"""MM-Vet v2 evaluation task.

MM-Vet v2 is a challenging benchmark to evaluate large multimodal models
for integrated capabilities across multiple dimensions:
- Recognition (rec)
- OCR (ocr)
- Knowledge (know)
- Language Generation (gen)
- Spatial Awareness (spat)
- Mathematics (math)
- Sequential Reasoning (seq)

The benchmark uses GPT-4 (gpt-4-0613) as a judge to score model outputs
on a 0.0-1.0 scale, evaluating whether predictions match ground truth
answers which may contain <AND> or <OR> logic for multiple acceptable elements.

Paper: "MM-Vet v2: A Challenging Benchmark to Evaluate Large Multimodal
Models for Integrated Capabilities" (arXiv:2408.00765)
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.mmvetv2 import get_mmvetv2_dataset
from openbench.metrics.mmvetv2 import mmvetv2_capability_metrics_list
from openbench.scorers.mmvetv2 import mmvetv2_scorer


@task
def mmvetv2(
    grader_model: str = "openai/gpt-4-0613",
    num_grading_attempts: int = 5,
) -> Task:
    """MM-Vet v2: Evaluating Large Multimodal Models for Integrated Capabilities.

    Evaluates vision-language models across 517 challenging questions requiring
    integrated capabilities. Uses GPT-4 as judge with few-shot prompting to
    score predictions on a 0.0-1.0 scale.

    NOTE: This evaluation uses LLM-as-a-judge with multi-run grading averaging.
    Each sample is graded {num_grading_attempts} times by GPT-4 to account for
    variance, and scores are averaged.

    Dataset: 517 questions (218 from v1, 299 new in v2)
    Capabilities: rec, ocr, know, gen, spat, math, seq
    Scoring: GPT-4 judge with 11-point scale (0.0-1.0)

    Args:
        grader_model: Model to use for grading (default: openai/gpt-4-0613)
        num_grading_attempts: Number of grading runs per sample (default: 5)
                              Set to 1 for faster single-run evaluation.
                              Paper uses 5 runs for published results.

    Example:
        # Default: 5 grading runs per sample (matches paper)
        bench eval mmvetv2 --model anthropic/claude-3.5-sonnet

        # Fast single-run evaluation (less accurate)
        bench eval mmvetv2 --model anthropic/claude-3.5-sonnet \\
            --T num_grading_attempts=1

        # Custom grader model
        bench eval mmvetv2 --model anthropic/claude-3.5-sonnet \\
            --T grader_model=openai/gpt-4o

    Returns:
        Task configured for MM-Vet v2 evaluation
    """
    dataset = get_mmvetv2_dataset()

    # Print helpful message about multi-run grading
    print(
        "💡 Tip: Use -T num_grading_attempts=N to customize number of averaged model-graded responses (default: 5)"
    )

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=mmvetv2_scorer(
            grader_model=grader_model,
            num_grading_attempts=num_grading_attempts,
        ),
        metrics=mmvetv2_capability_metrics_list(),
        name="mmvetv2",
    )
