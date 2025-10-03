"""
XCOPA: Cross-lingual Choice of Plausible Alternatives

XCOPA is a multilingual dataset for causal commonsense reasoning that evaluates
the ability of machine learning models to transfer commonsense reasoning across languages.
The dataset covers 11 languages from 11 families and several areas around the globe,
including resource-poor languages like Eastern Apurímac Quechua and Haitian Creole.

Languages:
- et: Estonian
- ht: Haitian Creole
- id: Indonesian
- it: Italian
- qu: Quechua
- sw: Swahili
- ta: Tamil
- th: Thai
- tr: Turkish
- vi: Vietnamese
- zh: Chinese

Sample usage:
```bash
bench eval xcopa_it --model "groq/llama-3.1-70b"
bench eval xcopa_sw --model "groq/llama-3.1-70b"
bench eval xcopa_zh --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{ponti2020xcopa,
    title={{XCOPA}: A Multilingual Dataset for Causal Commonsense Reasoning},
    author={Edoardo M. Ponti and Goran Glava\v{s} and Olga Majewska and Qianchu Liu and Ivan Vuli\'{c} and Anna Korhonen},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    pages={2362--2376},
    year={2020},
    url={https://ducdauge.github.io/files/xcopa.pdf}
}
"""

from inspect_ai.solver._multiple_choice import prompt
from inspect_ai import Task, task
from inspect_ai.solver import Choices
from openbench.utils.mcq import MCQEval, MCQSample

# MCQ Template
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

# Mapping of language codes to full names
XCOPA_LANGUAGES = {
    "et": "Estonian",
    "ht": "Haitian Creole",
    "id": "Indonesian",
    "it": "Italian",
    "qu": "Quechua",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert an XCOPA record to an OpenBench MCQSample.

    XCOPA tests causal reasoning by asking which of two alternatives
    is more plausibly the cause or effect of a given premise.

    Same structure as COPA but in 11 different languages.
    """
    connector = {"cause": "because", "effect": "therefore"}[record["question"]]
    premise = record["premise"].strip()
    if premise.endswith("."):
        premise = premise[:-1]

    # Format: "The man turned on the faucet therefore"
    input_question = f"{premise} {connector}"

    # Lowercase first letter of each choice
    choice1 = record["choice1"]
    choice2 = record["choice2"]
    choice1 = choice1[0].lower() + choice1[1:] if choice1 else choice1
    choice2 = choice2[0].lower() + choice2[1:] if choice2 else choice2

    input_msg = prompt(
        question=input_question,
        choices=Choices([choice1, choice2]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    int_to_char = {0: "A", 1: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
        metadata={
            "idx": record.get("idx"),
            "changed": record.get("changed", False),
            "question_type": record["question"],
        },
    )


# Main XCOPA function - all language tasks call this
@task
def xcopa(language: str = "it", split: str = "validation") -> Task:
    """
    Family benchmark for XCOPA - run any language by code.

    Args:
        language: Language code to evaluate. Available languages:
                  - et: Estonian
                  - ht: Haitian Creole
                  - id: Indonesian
                  - it: Italian (default)
                  - qu: Quechua
                  - sw: Swahili
                  - ta: Tamil
                  - th: Thai
                  - tr: Turkish
                  - vi: Vietnamese
                  - zh: Chinese
        split: Dataset split to use (default: "validation")
               Options: "validation" (100 samples), "test" (500 samples)

    Returns:
        Task: The specified XCOPA language task

    Sample usage:
    ```bash
    # Run Italian XCOPA
    bench eval xcopa --model "openrouter/openai/gpt-oss-120b" -M only=groq

    # Programmatic access to specific language
    from openbench.evals.xcopa import xcopa
    task = xcopa(language="zh", split="test")
    ```
    """
    if language not in XCOPA_LANGUAGES:
        available = ", ".join(XCOPA_LANGUAGES.keys())
        raise ValueError(f"Invalid XCOPA language '{language}'. Available: {available}")

    return MCQEval(
        name=f"xcopa_{language}",
        dataset_path="cambridgeltl/xcopa",
        subset_name=language,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )


# Individual language wrapper functions
@task
def xcopa_et(split: str = "validation") -> Task:
    """XCOPA: Estonian"""
    return xcopa(language="et", split=split)


@task
def xcopa_ht(split: str = "validation") -> Task:
    """XCOPA: Haitian Creole"""
    return xcopa(language="ht", split=split)


@task
def xcopa_id(split: str = "validation") -> Task:
    """XCOPA: Indonesian"""
    return xcopa(language="id", split=split)


@task
def xcopa_it(split: str = "validation") -> Task:
    """XCOPA: Italian"""
    return xcopa(language="it", split=split)


@task
def xcopa_qu(split: str = "validation") -> Task:
    """XCOPA: Quechua"""
    return xcopa(language="qu", split=split)


@task
def xcopa_sw(split: str = "validation") -> Task:
    """XCOPA: Swahili"""
    return xcopa(language="sw", split=split)


@task
def xcopa_ta(split: str = "validation") -> Task:
    """XCOPA: Tamil"""
    return xcopa(language="ta", split=split)


@task
def xcopa_th(split: str = "validation") -> Task:
    """XCOPA: Thai"""
    return xcopa(language="th", split=split)


@task
def xcopa_tr(split: str = "validation") -> Task:
    """XCOPA: Turkish"""
    return xcopa(language="tr", split=split)


@task
def xcopa_vi(split: str = "validation") -> Task:
    """XCOPA: Vietnamese"""
    return xcopa(language="vi", split=split)


@task
def xcopa_zh(split: str = "validation") -> Task:
    """XCOPA: Chinese"""
    return xcopa(language="zh", split=split)
