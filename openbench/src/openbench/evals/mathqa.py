"""
MathQA - Mathematical Question Answering

MathQA is a large-scale dataset of math word problems with multiple-choice
answers. The dataset contains 37,297 problems covering various mathematical
concepts. Each problem includes a natural language question, multiple-choice
options, the correct answer, and a rationale explaining the solution.

Sample usage:
```bash
bench eval math_qa --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{amini2019mathqa,
    title={MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms},
    author={Amini, Aida and Gabriel, Saadia and Lin, Shanchuan and Koncel-Kedziorski, Rik and Choi, Yejin and Hajishirzi, Hannaneh},
    booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    pages={2357--2367},
    year={2019}
}
"""

import re
from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def parse_mathqa_options(options_str: str) -> list[str]:
    """
    Parse MathQA options string into a list of answer choices.

    The options are formatted as: "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
    We need to extract the values after each letter.

    Args:
        options_str: String containing the formatted options

    Returns:
        List of answer choice strings
    """
    # Pattern to match: letter ) value
    # This handles various formats including numbers, decimals, and text
    pattern = r"[a-e]\s*\)\s*([^,]+?)(?=\s*,\s*[a-e]\s*\)|$)"
    matches = re.findall(pattern, options_str, re.IGNORECASE)

    # Clean up the matches (strip whitespace)
    choices = [match.strip() for match in matches]

    return choices


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a MathQA record to an OpenBench MCQSample."""
    question = record["Problem"]

    # Parse the options string into a list of choices
    options_str = record["options"]
    choices = parse_mathqa_options(options_str)

    # Ensure we have valid choices
    if len(choices) < 2:
        # Fallback: if parsing failed, try simple split
        choices = [opt.strip() for opt in options_str.split(",")]

    # Create the multiple choice prompt
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    # Get the correct answer (already a letter like 'a', 'b', 'c')
    correct_letter = record["correct"].strip().upper()

    return MCQSample(
        input=prompt,
        target=correct_letter,
        metadata={
            "question": question,
            "category": record.get("category", ""),
            "rationale": record.get("Rationale", ""),
            "annotated_formula": record.get("annotated_formula", ""),
        },
    )


@task
def math_qa(split: str = "validation") -> Task:
    """
    Evaluate the MathQA benchmark for mathematical reasoning.

    MathQA contains math word problems with multiple-choice answers, designed
    to test mathematical reasoning and problem-solving abilities. Each problem
    includes a rationale and annotated formula for the solution.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for MathQA evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="math_qa",
        dataset_path="allenai/math_qa",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        dataset_kwargs={"trust": True},
    )
