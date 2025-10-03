"""
WSC273: Winograd Schema Challenge (273 problems)

This benchmark evaluates models on pronoun resolution requiring commonsense reasoning.
Each problem presents a sentence with an ambiguous pronoun and asks which noun phrase
the pronoun refers to. The challenge is that resolving the pronoun requires understanding
the world and cannot be done with simple syntactic patterns.

Dataset: lighteval/winograd_wsc, subset wsc273
Split: test (273 samples)

Reference: Levesque et al. (2012) - The Winograd Schema Challenge

Sample usage:
```bash
bench eval wsc273 --model groq/llama-3.1-70b-versatile
```
"""

from inspect_ai import Task, task
from inspect_ai.solver import Choices
from inspect_ai.solver._multiple_choice import prompt
from openbench.utils.mcq import MCQEval, MCQSample


WINOGRAD_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a WSC273 record to an OpenBench MCQSample.

    The dataset contains:
    - text: Sentence with ambiguous pronoun
    - pronoun: The pronoun to resolve
    - quote: Text snippet containing the pronoun
    - options: List of two candidate noun phrases
    - label: Index (0 or 1) of correct referent
    """
    text = record["text"]
    pronoun = record["pronoun"]
    options = record["options"]

    # Create question
    question = f"{text}\n\nWhat does '{pronoun}' refer to?"

    # Create MCQ with the two options
    input_msg = prompt(
        question=question,
        choices=Choices(options),
        template=str(WINOGRAD_TEMPLATE),
    )

    # Convert label index to letter (0->A, 1->B)
    label_idx = record["label"]
    target = chr(65 + label_idx)  # 65 is ASCII for 'A'

    return MCQSample(
        input=input_msg,
        target=target,
        metadata={
            "source": record.get("source", ""),
            "pronoun": pronoun,
        },
    )


@task
def wsc273(split: str = "test") -> Task:
    """Evaluate the Winograd Schema Challenge (273 problems)."""
    return MCQEval(
        name="wsc273",
        dataset_path="lighteval/winograd_wsc",
        subset_name="wsc273",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )
