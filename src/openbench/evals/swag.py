"""
SWAG - Situations With Adversarial Generations

SWAG is a large-scale dataset for grounded commonsense inference, unifying natural
language inference and physically grounded reasoning. The dataset consists of 113k
multiple choice questions about grounded situations, where each question is a video
caption with four answer choices about what might happen next in the scene.

Sample usage:
```bash
bench eval swag --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{zellers2018swag,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    year={2018}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a SWAG record to an OpenBench MCQSample."""
    # The startphrase contains the context (or we can use sent1 + sent2)
    # Using sent1 and sent2 to construct the question
    question = f"{record['sent1']} {record['sent2']}"

    # Four possible endings
    choices = [
        record["ending0"],
        record["ending1"],
        record["ending2"],
        record["ending3"],
    ]

    # Create the multiple choice prompt
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    # Convert label to letter (0 -> A, 1 -> B, etc.)
    # Handle test set where label might be -1 (no ground truth)
    if record["label"] == -1:
        # For test set, we don't have labels, use A as placeholder
        target = "A"
    else:
        target = chr(65 + int(record["label"]))

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "video_id": record.get("video-id", "unknown"),
            "startphrase": record.get("startphrase", ""),
            "gold_source": record.get("gold-source", "unknown"),
        },
    )


@task
def swag(split: str = "validation") -> Task:
    """
    Evaluate the SWAG benchmark for grounded commonsense reasoning.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for SWAG evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="swag",
        dataset_path="swag",
        subset_name="regular",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )
