"""
ANLI - Adversarial Natural Language Inference

The Adversarial Natural Language Inference (ANLI) benchmark is a large-scale NLI dataset
collected via an iterative, adversarial human-and-model-in-the-loop procedure.
ANLI contains three rounds of progressively harder examples:
- Round 1 (R1): Easiest, collected against BERT-based models
- Round 2 (R2): Medium difficulty, collected against RoBERTa-based models
- Round 3 (R3): Hardest, collected against advanced models including ALBERT and XLNet

Each example consists of a premise and hypothesis, with three-way classification:
- 0: entailment
- 1: neutral
- 2: contradiction

Sample usage:
```bash
bench eval anli_r1 --model "groq/llama-3.1-8b-instant"
bench eval anli_r2 --model "groq/llama-3.1-8b-instant"
bench eval anli_r3 --model "groq/llama-3.1-8b-instant"
bench eval anli --subset r1 --model "groq/llama-3.1-8b-instant"
```

Citation:
@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial NLI: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and Williams, Adina  and Dinan, Emily  and Bansal, Mohit  and Weston, Jason  and Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
}
"""

from inspect_ai import Task, task
from inspect_ai.solver import Choices
from inspect_ai.solver._multiple_choice import prompt
from openbench.utils.mcq import MCQEval, MCQSample

# MCQ Template
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


def record_to_mcq_sample_anli(record: dict) -> MCQSample:
    """Convert an ANLI record to an OpenBench MCQSample.

    ANLI tests natural language inference with three-way classification:
    entailment, neutral, and contradiction.
    """
    input_question = (
        f"Premise: {record['premise']}\n"
        f"Hypothesis: {record['hypothesis']}\n"
        f"Question: What is the relationship between the premise and hypothesis?"
    )

    input_msg = prompt(
        question=input_question,
        choices=Choices(["entailment", "neutral", "contradiction"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 0 = entailment (A), Label 1 = neutral (B), Label 2 = contradiction (C)
    int_to_char = {0: "A", 1: "B", 2: "C"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
        id=record.get("uid", None),
    )


# Mapping of round names to their configuration
ANLI_ROUND_CONFIG = {
    "r1": {
        "task_name": "anli_r1",
        "split": "test_r1",
    },
    "r2": {
        "task_name": "anli_r2",
        "split": "test_r2",
    },
    "r3": {
        "task_name": "anli_r3",
        "split": "test_r3",
    },
}


# Main ANLI function - all round tasks call this
@task
def anli(round: str = "r1") -> Task:
    """
    Family benchmark for ANLI - run any ANLI round by name.

    Args:
        round: ANLI round to evaluate. Available rounds:
               - r1: Round 1 (easiest, BERT-level)
               - r2: Round 2 (medium, RoBERTa-level)
               - r3: Round 3 (hardest, ALBERT/XLNet-level)

    Returns:
        Task: The specified ANLI round task
    """
    if round not in ANLI_ROUND_CONFIG:
        available = ", ".join(ANLI_ROUND_CONFIG.keys())
        raise ValueError(f"Invalid ANLI round '{round}'. Available: {available}")

    config = ANLI_ROUND_CONFIG[round]

    return MCQEval(
        name=config["task_name"],
        dataset_path="facebook/anli",
        subset_name=None,  # No subset name needed for ANLI
        record_to_mcq_sample=record_to_mcq_sample_anli,
        split=config["split"],
        auto_id=False,  # Use the uid from the dataset
    )


# Individual task functions - convenience wrappers that call anli(round=...)
@task
def anli_r1() -> Task:
    """
    ANLI Round 1 - Easiest round

    Round 1 examples were collected against BERT-based models and represent
    the baseline difficulty level of adversarial NLI examples.

    Returns:
        Task: ANLI Round 1 evaluation task
    """
    return anli(round="r1")


@task
def anli_r2() -> Task:
    """
    ANLI Round 2 - Medium difficulty

    Round 2 examples were collected against RoBERTa-based models and are
    more challenging than Round 1.

    Returns:
        Task: ANLI Round 2 evaluation task
    """
    return anli(round="r2")


@task
def anli_r3() -> Task:
    """
    ANLI Round 3 - Hardest round

    Round 3 examples were collected against advanced models including ALBERT
    and XLNet, representing the most challenging adversarial NLI examples.

    Returns:
        Task: ANLI Round 3 evaluation task
    """
    return anli(round="r3")
