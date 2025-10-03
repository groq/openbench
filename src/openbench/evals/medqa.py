"""
MedQA: A Free and Open Medical Question Answering Dataset

MedQA is a medical question answering dataset based on the USMLE (United States Medical Licensing Examination)
exam questions. The dataset includes questions with 4-5 answer options covering various medical topics.

Sample usage:
```bash
bench eval medqa --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@article{jin2021disease,
    title={What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
    author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
    journal={Applied Sciences},
    volume={11},
    number={14},
    pages={6421},
    year={2021},
    publisher={MDPI}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a MedQA record to an OpenBench MCQSample.

    MedQA provides USMLE-style medical questions with 4-5 answer options.
    The dataset uses a simplified structure with:
    - question: Question text
    - options: Dictionary with option keys (A, B, C, D, E)
    - answer_idx: Letter of correct answer (A-E)
    """
    question = record["question"]

    # Get options - they should be in a dict or list format
    # Handle different possible formats
    if "options" in record:
        options_data = record["options"]
        if isinstance(options_data, dict):
            # Options as dict with keys A, B, C, D, E
            option_keys = sorted([k for k in options_data.keys() if k in "ABCDE"])
            options = [options_data[k] for k in option_keys]
        elif isinstance(options_data, list):
            # Options as list
            options = options_data
        else:
            raise ValueError(f"Unexpected options format: {type(options_data)}")
    else:
        # Fallback: look for individual option fields
        options = []
        for letter in "ABCDE":
            if letter in record:
                options.append(record[letter])
            elif letter.lower() in record:
                options.append(record[letter.lower()])

    # Create prompt with dynamic number of options
    prompt = create_dynamic_multiple_choice_prompt(question, options)

    # Get correct answer
    if "answer_idx" in record:
        target = record["answer_idx"].upper()
    elif "answer" in record:
        # If answer is text, find which option it matches
        answer_text = record["answer"]
        try:
            answer_idx = options.index(answer_text)
            target = chr(65 + answer_idx)
        except ValueError:
            # If exact match fails, use first letter of answer if it's A-E
            if answer_text and answer_text[0].upper() in "ABCDE":
                target = answer_text[0].upper()
            else:
                raise ValueError(f"Cannot determine target from answer: {answer_text}")
    else:
        raise ValueError("No answer field found in record")

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "question_id": record.get("id", record.get("qid", "unknown")),
        },
    )


@task
def medqa(split: str = "test") -> Task:
    """
    Evaluate the MedQA dataset.

    MedQA contains USMLE-style medical exam questions with 4-5 answer options.
    Questions cover a wide range of medical knowledge required for medical licensing.

    Args:
        split: Dataset split to use (default: "test")
               Options: "train", "validation", "test"

    Sample usage:
    ```bash
    bench eval medqa --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 100
    ```
    """
    return MCQEval(
        name="medqa",
        dataset_path="GBaker/MedQA-USMLE-4-options",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )
