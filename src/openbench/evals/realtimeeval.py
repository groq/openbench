from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.realtimeeval import get_dataset
from openbench.scorers.simpleqa import simpleqa_scorer


@task
def realtimeeval(
    grader_model: str = "openai/gpt-4.1-2025-04-14",
    json_file: str = "realtimeeval_questions.json"
) -> Task:
    """RealtimeEval: A customizable evaluation that loads questions from a local JSON file.

    This eval copies the simpleQA implementation exactly but loads questions and answers
    from a local JSON file instead of a remote CSV. Uses the same model-based grading
    to assess factual accuracy of responses.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)
        json_file: Path to the JSON file containing questions and answers

    Returns:
        Task configured for RealtimeEval evaluation
    """
    return Task(
        dataset=get_dataset(json_file=json_file),
        solver=[generate()],
        scorer=simpleqa_scorer(model=grader_model),
        name="realtimeeval",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for factual QA
        ),
    )