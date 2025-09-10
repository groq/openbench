from typing import Any
import json

from inspect_ai.dataset import Sample, Dataset, hf_dataset

ARC_AGI_SYSTEM_PROMPT = """
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:
""".strip()


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    Convert an ARC-AGI HuggingFace dataset record to a Sample for evaluation.

    Args:
        record: Dictionary containing 'train', 'test', and 'filename' fields

    Returns:
        Sample: Converted sample with prompt and expected output
    """
    # Extract training examples and format them
    training_examples = ""
    for i, pair in enumerate(record["train"]):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += json.dumps(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += json.dumps(pair["output"]) + "\n\n"

    # Extract test input (assuming single test case per task)
    test_input = json.dumps(record["test"][0]["input"])
    test_output = record["test"][0]["output"]

    # Format the complete prompt
    prompt = ARC_AGI_SYSTEM_PROMPT.format(
        training_examples=training_examples, test_input=test_input
    )

    return Sample(
        id=record["filename"],
        input=prompt,
        target=json.dumps(test_output),
        metadata={
            "train": record["train"],
            "test": record["test"],
            "filename": record["filename"],
            "expected_grid": test_output,
        },
    )


def get_arc_agi_dataset(version: int = 1) -> Dataset:
    """
    Load the ARC-AGI evaluation dataset from HuggingFace.

    Args:
        version: Version of ARC-AGI dataset (1 or 2)

    Returns:
        Dataset: The ARC-AGI evaluation dataset
    """
    # To change the dataset, adjust the HuggingFace path here
    if version == 1:
        path = "lee64/arc-agi-evaluation"
    elif version == 2:
        path = "lee64/arc-agi-2-evaluation"
    else:
        raise ValueError(f"Unknown ARC-AGI version: {version}. Must be 1 or 2.")

    return hf_dataset(
        path=path,
        split="test",
        sample_fields=record_to_sample,
    )
