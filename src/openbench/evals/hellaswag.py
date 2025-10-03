"""
HellaSwag - Commonsense Inference in Natural Language

HellaSwag is a challenge dataset for evaluating commonsense NLI. It tests a
model's ability to complete a sentence with a coherent continuation given
context from ActivityNet captions and WikiHow articles.

Sample usage:
```bash
bench eval hellaswag --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""

import re
from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def hellaswag_preprocess(text: str) -> str:
    """
    Preprocess HellaSwag text by removing WikiHow artifacts and cleaning formatting.

    From LM Eval Harness preprocessing.
    """
    # Remove [title] markers from WikiHow data
    text = text.replace(" [title]", ". ")

    # Remove all bracketed content
    text = re.sub(r"\[.*?\]", "", text)

    # Normalize spaces
    text = text.replace("  ", " ")

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a HellaSwag record to an OpenBench MCQSample."""
    # Build context from ctx_a and ctx_b (capitalize ctx_b)
    ctx = f"{record['ctx_a']} {record['ctx_b'].capitalize()}"

    # Create the question with activity label
    question = f"{record['activity_label']}: {ctx}"

    # Preprocess question and endings
    question = hellaswag_preprocess(question)
    endings = [hellaswag_preprocess(ending) for ending in record["endings"]]

    # Create prompt
    prompt = create_dynamic_multiple_choice_prompt(question, endings)

    # Convert label to letter (0 -> A, 1 -> B, etc.)
    # Handle test set where label is empty string
    if record["label"] == "":
        # For test set, we don't have labels, use A as placeholder
        # These samples will be filtered out if evaluation_splits doesn't include test
        target = "A"
    else:
        target = chr(65 + int(record["label"]))

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "activity_label": record["activity_label"],
            "split_type": record.get("split_type", "unknown"),
            "source": record.get("source_id", "unknown"),
        },
        id=record.get("ind", None),
    )


@task
def hellaswag(split: str = "validation") -> Task:
    """
    Evaluate the HellaSwag benchmark for commonsense NLI.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for HellaSwag evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="hellaswag",
        dataset_path="hellaswag",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )
