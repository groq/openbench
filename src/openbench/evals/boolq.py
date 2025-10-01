"""
BoolQ: A Question Answering Dataset for Boolean Reasoning
https://arxiv.org/abs/1905.10044

Sample usage:
```bash
uv run openbench eval boolq --model "groq/llama-3.1-8b-instant"
```
The prompt is based on the default prompt (default.yaml) from
the EleutherAI/lm-evaluation-harness implementation:

https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/super_glue/boolq/default.yaml
"""

from inspect_ai.solver._multiple_choice import prompt
from inspect_ai import Task, task
from inspect_ai.solver import Choices
from openbench.utils.mcq import MCQEval, MCQSample

# ---------- MCQ ABSTRACTION -----------
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


# original implementation uses built in multiple_choice solver
def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a BoolQ record to an OpenBench MCQSample."""
    answer_label = record["answer"]
    int_to_char = {0: "A", 1: "B"}

    passage = record["passage"]
    question = record["question"]

    input_question = f"{passage}\nQuestion: {question}?\nAnswer:"

    input_msg = prompt(
        question=input_question,
        choices=Choices(["false", "true"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    return MCQSample(
        input=input_msg,
        target=int_to_char[answer_label],
    )


@task
def boolq(split="validation") -> Task:
    """Evaluate the BoolQ dataset. MCQ Abstracted."""
    return MCQEval(
        name="boolq",
        dataset_path="boolq",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )
