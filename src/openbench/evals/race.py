"""
RACE - ReAding Comprehension from Examinations

RACE is a large-scale reading comprehension dataset with passages and questions
from English exams for Chinese students in middle and high school. The dataset
contains over 28,000 passages and nearly 100,000 questions.

This implementation includes RACE-High (high school level passages).

Dataset: EleutherAI/race
Paper: https://arxiv.org/abs/1704.04683

Sample usage:
```bash
bench eval race_high --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@inproceedings{lai2017race,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
    pages={785--794},
    year={2017}
}
"""

import ast
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from openbench.utils.mcq import MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_samples(record: dict) -> list[Sample]:
    """Convert a RACE record to multiple OpenBench MCQSamples.

    The RACE dataset has one article with multiple problems (questions).
    Each problem becomes a separate sample.

    Args:
        record: Dictionary containing:
            - article: The reading passage
            - problems: String representation of list of questions, each having:
                - question: The question text
                - options: List of answer choices
                - answer: The correct answer letter (A, B, C, or D)

    Returns:
        List of MCQSamples, one per problem
    """
    article = record["article"]
    # Use ast.literal_eval since the string uses single quotes (Python repr)
    problems = ast.literal_eval(record["problems"])

    samples: list[Sample] = []
    for problem in problems:
        question = problem["question"]
        options = problem["options"]
        answer = problem["answer"]

        # Create the full prompt: passage + question + options
        full_question = f"Passage:\n{article}\n\nQuestion: {question}"
        prompt = create_dynamic_multiple_choice_prompt(full_question, options)

        samples.append(
            MCQSample(
                input=prompt,
                target=answer,
                metadata={
                    "example_id": record.get("example_id", ""),
                },
            )
        )

    return samples


@task
def race_high(split: str = "test") -> Task:
    """
    Evaluate RACE-High reading comprehension benchmark.

    RACE-High contains passages from high school English exams, typically
    more challenging than RACE-Middle. Each passage is followed by
    multiple questions with 4 answer choices each.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "test" (standard evaluation split)

    Returns:
        Task: Inspect AI task for RACE-High evaluation
    """
    from inspect_ai.dataset import hf_dataset
    from inspect_ai.solver import generate
    from inspect_ai.model import GenerateConfig
    from openbench.scorers.mcq import create_mcq_scorer

    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    # Load dataset and expand samples (one article -> multiple questions)
    dataset = hf_dataset(
        "EleutherAI/race",
        split=split,
        sample_fields=record_to_samples,
        auto_id=True,
        name="high",
    )

    return Task(
        name="race_high",
        dataset=dataset,
        solver=[generate()],
        scorer=create_mcq_scorer()(),
        config=GenerateConfig(),
    )
