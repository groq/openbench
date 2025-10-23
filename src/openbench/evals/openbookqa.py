"""openbench implementation of OpenBookQA (MCQ abstracted)."""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record) -> MCQSample:
    """Convert an OpenBookQA record to an openbench MCQSample."""
    question = record["question_stem"]
    options = [choice for choice in record["choices"]["text"]]
    prompt = create_dynamic_multiple_choice_prompt(question, options)

    return MCQSample(
        input=prompt,
        target=record["answerKey"],
        id=record.get("id"),
        metadata={
            "choice_labels": record["choices"]["label"],
        },
    )


@task
def openbookqa(split: str = "validation") -> Task:
    """OpenBookQA multiple choice science question evaluation (MCQ Abstracted)."""
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="openbookqa",
        dataset_path="allenai/openbookqa",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        dataset_kwargs={},
    )
