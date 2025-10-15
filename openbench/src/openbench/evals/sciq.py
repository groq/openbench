"""
SciQ - Science Questions

SciQ is a dataset of crowdsourced science exam questions covering Physics,
Chemistry, Biology, and other scientific domains. The dataset contains 13,679
multiple-choice questions with 4 answer options each. Most questions include
a supporting paragraph with evidence for the correct answer.

Sample usage:
```bash
bench eval sciq --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{welbl2017crowdsourcing,
    title={Crowdsourcing Multiple Choice Science Questions},
    author={Welbl, Johannes and Liu, Nelson F. and Gardner, Matt},
    booktitle={Proceedings of the 3rd Workshop on Noisy User-generated Text},
    pages={94--106},
    year={2017}
}
"""

import random
from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a SciQ record to an OpenBench MCQSample."""
    question = record["question"]

    # Collect all answer choices
    choices = [
        record["correct_answer"],
        record["distractor1"],
        record["distractor2"],
        record["distractor3"],
    ]

    # Shuffle choices and track correct answer position
    # We need to maintain consistency, so we'll use the question as seed
    shuffled_indices = list(range(4))
    random.Random(question).shuffle(shuffled_indices)

    # Apply the shuffle
    shuffled_choices = [choices[i] for i in shuffled_indices]

    # Find where the correct answer ended up (index 0 was the correct one)
    correct_position = shuffled_indices.index(0)
    target = chr(65 + correct_position)  # Convert to letter (A, B, C, D)

    # Create the multiple choice prompt
    prompt = create_dynamic_multiple_choice_prompt(question, shuffled_choices)

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "question": question,
            "support": record.get("support", ""),
        },
    )


@task
def sciq(split: str = "validation") -> Task:
    """
    Evaluate the SciQ benchmark for science question answering.

    SciQ contains crowdsourced science exam questions with 4 multiple-choice
    options covering various scientific domains including Physics, Chemistry,
    and Biology.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for SciQ evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="sciq",
        dataset_path="allenai/sciq",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )
