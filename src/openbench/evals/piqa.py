"""
PIQA - Physical Interaction QA

PIQA tests a model's ability to reason about physical commonsense by choosing
the more appropriate solution to achieve a physical goal. The dataset consists
of goal-solution pairs where the correct solution demonstrates understanding of
basic physical interactions and common sense.

Sample usage:
```bash
bench eval piqa --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{bisk2020piqa,
    title={PIQA: Reasoning about Physical Commonsense in Natural Language},
    author={Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={34},
    pages={7432--7439},
    year={2020}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a PIQA record to an OpenBench MCQSample."""
    # The goal is the question/physical scenario
    question = record["goal"]

    # Two solution options
    choices = [record["sol1"], record["sol2"]]

    # Create the multiple choice prompt
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    # Convert label to letter (0 -> A, 1 -> B)
    # Handle test set where label is -1 (no ground truth)
    if record["label"] == -1:
        # For test set, we don't have labels, use A as placeholder
        target = "A"
    else:
        target = chr(65 + int(record["label"]))

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "goal": record["goal"],
        },
    )


@task
def piqa(split: str = "validation") -> Task:
    """
    Evaluate the PIQA benchmark for physical commonsense reasoning.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for PIQA evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="piqa",
        dataset_path="ybisk/piqa",
        subset_name="plain_text",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )
