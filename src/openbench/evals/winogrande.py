"""
Winogrande - Pronoun Resolution and Commonsense Reasoning

Winogrande is a large-scale dataset for commonsense reasoning and pronoun resolution.
It presents sentences with a blank (represented by "_") and two options to fill in
the blank. The model must use commonsense reasoning to determine which option makes
more sense in context.

Sample usage:
```bash
bench eval winogrande --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{sakaguchi2019winogrande,
    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2020}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a Winogrande record to an OpenBench MCQSample."""
    # Split the sentence at the blank (_)
    parts = record["sentence"].split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '_' in sentence: {record['sentence']}")

    question_prefix = parts[0].strip()
    question_suffix = parts[1].strip()

    # Create the question by combining prefix, blank indicator, and suffix
    question = f"{question_prefix} _____ {question_suffix}"

    # The two options to fill in the blank
    options = [record["option1"], record["option2"]]

    # Create prompt
    prompt = create_dynamic_multiple_choice_prompt(question, options)

    # Convert answer (1 or 2) to letter (A or B)
    # Handle test set where answer might be empty string
    if record["answer"] == "":
        # For test set, we don't have labels, use A as placeholder
        target = "A"
    else:
        answer_idx = int(record["answer"]) - 1  # Convert 1-indexed to 0-indexed
        target = chr(65 + answer_idx)  # Convert to letter: 0 -> A, 1 -> B

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "sentence": record["sentence"],
            "option1": record["option1"],
            "option2": record["option2"],
        },
    )


@task
def winogrande(split: str = "validation") -> Task:
    """
    Evaluate the Winogrande benchmark for pronoun resolution and commonsense reasoning.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for Winogrande evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="winogrande",
        dataset_path="winogrande",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        subset_name="winogrande_xl",
    )
