"""
XWinograd - Multilingual Winograd Schema Challenge

XWinograd is a multilingual version of the Winograd Schema Challenge, testing
commonsense reasoning and coreference resolution across multiple languages.

The dataset contains sentences with ambiguous pronouns that require commonsense
reasoning to resolve. Models must choose which noun phrase the pronoun refers to.

Available languages:
- English (en): 2325 samples
- French (fr): 83 samples
- Japanese (jp): 959 samples
- Portuguese (pt): 263 samples
- Russian (ru): 315 samples
- Chinese (zh): 504 samples

Sample usage:
```bash
bench eval xwinograd_en --model "groq/llama-3.1-70b"
bench eval xwinograd_fr --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{tikhonov-ryabinin-2021-heads,
    title = "It{'}s All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning",
    author = "Tikhonov, Alexey and Ryabinin, Max",
    booktitle = "Findings of ACL 2021",
    year = "2021"
}
"""

from inspect_ai import Task, task
from inspect_ai.solver._multiple_choice import prompt
from inspect_ai.solver import Choices
from openbench.utils.mcq import MCQEval, MCQSample

# Language mapping
XWINOGRAD_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "jp": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}

XWINOGRAD_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert an XWinograd record to an OpenBench MCQSample."""
    sentence = record["sentence"]
    option1 = record["option1"]
    option2 = record["option2"]
    answer = record["answer"]  # "1" or "2"

    # Create prompt using Inspect AI's prompt helper
    input_msg = prompt(
        question=sentence,
        choices=Choices([option1, option2]),
        template=str(XWINOGRAD_TEMPLATE),
    )

    # Convert answer "1" or "2" to "A" or "B"
    int_to_char = {"1": "A", "2": "B"}
    target = int_to_char[answer]

    return MCQSample(
        input=input_msg,
        target=target,
    )


@task
def xwinograd(language: str = "en", split: str = "test") -> Task:
    """
    Family benchmark for XWinograd - run any language by code.

    Args:
        language: Language code to evaluate (default: "en")
                 Options: en, fr, jp, pt, ru, zh
        split: Dataset split to use (default: "test")

    Returns:
        Task: Inspect AI task for XWinograd evaluation
    """
    if language not in XWINOGRAD_LANGUAGES:
        available = ", ".join(XWINOGRAD_LANGUAGES.keys())
        raise ValueError(
            f"Invalid language '{language}'. Available languages: {available}"
        )

    return MCQEval(
        name=f"xwinograd_{language}",
        dataset_path="Muennighoff/xwinograd",
        subset_name=language,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )


# Individual wrapper functions for each language
@task
def xwinograd_en(split: str = "test") -> Task:
    """XWinograd: English (2325 samples)"""
    return xwinograd(language="en", split=split)


@task
def xwinograd_fr(split: str = "test") -> Task:
    """XWinograd: French (83 samples)"""
    return xwinograd(language="fr", split=split)


@task
def xwinograd_jp(split: str = "test") -> Task:
    """XWinograd: Japanese (959 samples)"""
    return xwinograd(language="jp", split=split)


@task
def xwinograd_pt(split: str = "test") -> Task:
    """XWinograd: Portuguese (263 samples)"""
    return xwinograd(language="pt", split=split)


@task
def xwinograd_ru(split: str = "test") -> Task:
    """XWinograd: Russian (315 samples)"""
    return xwinograd(language="ru", split=split)


@task
def xwinograd_zh(split: str = "test") -> Task:
    """XWinograd: Chinese (504 samples)"""
    return xwinograd(language="zh", split=split)
