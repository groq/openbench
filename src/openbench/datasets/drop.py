"""DROP dataset loader for Inspect AI."""

import gzip
import json
import random
from io import BytesIO
from urllib.request import urlopen

from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def record_to_sample(record: dict) -> Sample:
    """Convert a DROP record to an Inspect Sample."""
    # Format the context and question
    context = record["context"]
    completion = record["completion"]

    # Format input as context + completion (which contains the question)
    input_text = f"{context}\n\n{completion}"

    # Get reference answers (can be multiple, separated by |)
    target = record.get("ref_text", "")

    return Sample(
        input=input_text,
        target=target,
        metadata={
            "context": context,
            "completion": completion,
            "ref_text": target,
        },
    )


def get_dataset(
    num_examples: int | None = None,
    train_samples_per_prompt: int = 3,
    seed: int = 42,
) -> Dataset:
    """Load the DROP dataset.

    Args:
        num_examples: Number of examples to use (None for all)
        train_samples_per_prompt: Number of training examples for few-shot prompting
        seed: Random seed for sampling

    Returns:
        Dataset ready for evaluation
    """
    # URLs for the DROP dataset
    train_url = (
        "https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_train.jsonl.gz"
    )
    test_url = (
        "https://openaipublic.blob.core.windows.net/simple-evals/drop_v0_dev.jsonl.gz"
    )

    # Load training samples for few-shot examples
    with gzip.GzipFile(fileobj=BytesIO(urlopen(train_url).read()), mode="rb") as f:
        train_samples = [json.loads(line) for line in f.readlines()]

    # Load test samples
    with gzip.GzipFile(fileobj=BytesIO(urlopen(test_url).read()), mode="rb") as f:
        test_samples = [json.loads(line) for line in f.readlines()]

    # Sample if requested
    if num_examples:
        rng = random.Random(seed)
        test_samples = rng.sample(test_samples, min(num_examples, len(test_samples)))

    # Convert to Inspect samples
    samples = []
    rng = random.Random(seed)

    for test_sample in test_samples:
        # Get few-shot examples
        few_shot_examples = rng.sample(train_samples, train_samples_per_prompt)

        # Build the prompt with few-shot examples
        prompt_parts = [
            "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.",
            "\n# Examples",
        ]

        # Add few-shot examples
        for example in few_shot_examples:
            prompt_parts.append("\n---")
            prompt_parts.append(example["context"])
            prompt_parts.append(example["completion"])

        # Add the test example
        prompt_parts.append("\n# Your Task\n---")
        prompt_parts.append(test_sample["context"])
        prompt_parts.append(
            '\nThink step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.'
        )

        # Create the sample
        sample = Sample(
            input="\n".join(prompt_parts),
            target=test_sample.get("ref_text", ""),
            metadata={
                "context": test_sample["context"],
                "completion": test_sample["completion"],
                "ref_text": test_sample.get("ref_text", ""),
                "train_samples": few_shot_examples,
            },
        )
        samples.append(sample)

    return MemoryDataset(samples=samples, name="drop")
