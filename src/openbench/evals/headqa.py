"""
HeadQA: Healthcare Exam Questions from Spain

HeadQA is a multi-choice question answering test to encourage research on complex reasoning
based on graduate-level questions about medicine, nursing, psychology, chemistry, pharmacology, and biology.

Languages:
- en: English
- es: Spanish

Sample usage:
```bash
bench eval headqa_en --model "openrouter/openai/gpt-oss-120b" -M only=groq
bench eval headqa_es --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and
      G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    pages = "960--966",
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt

# Mapping of language codes to full names
HEADQA_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a HeadQA record to an OpenBench MCQSample.

    HeadQA provides healthcare exam questions with 4-5 answer options.
    The dataset includes:
    - qtext: Question text
    - answers: List of answer options with aid and atext
    - ra: Index of correct answer (1-based)
    - category: Subject area (biology, medicine, nursing, etc.)
    """
    question = record["qtext"]
    answers_list = record["answers"]

    # Extract answer texts in order
    # answers_list is a list of dicts with 'aid' and 'atext'
    # Sort by aid to ensure correct order
    sorted_answers = sorted(answers_list, key=lambda x: x["aid"])
    options = [answer["atext"] for answer in sorted_answers]

    # Create prompt with dynamic number of options
    prompt = create_dynamic_multiple_choice_prompt(question, options)

    # Convert 1-based index to 0-based, then to letter
    correct_index = record["ra"] - 1  # ra is 1-based
    target = chr(65 + correct_index)  # 0->A, 1->B, etc.

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "category": record.get("category", "unknown"),
            "year": record.get("year", "unknown"),
            "qid": record.get("qid"),
        },
    )


# Main HeadQA function - all language tasks call this
@task
def headqa(language: str = "en", split: str = "test") -> Task:
    """
    Family benchmark for HeadQA - run any language by code.

    Args:
        language: Language code to evaluate. Available languages:
                  - en: English (default)
                  - es: Spanish
        split: Dataset split to use (default: "test")
               Options: "train", "validation", "test"

    Returns:
        Task: The specified HeadQA language task

    Sample usage:
    ```bash
    # Run English HeadQA
    bench eval headqa --model "openrouter/openai/gpt-oss-120b" -M only=groq

    # Programmatic access to specific language
    from openbench.evals.headqa import headqa
    task = headqa(language="es", split="test")
    ```
    """
    if language not in HEADQA_LANGUAGES:
        available = ", ".join(HEADQA_LANGUAGES.keys())
        raise ValueError(
            f"Invalid HeadQA language '{language}'. Available: {available}"
        )

    return MCQEval(
        name=f"headqa_{language}",
        dataset_path="head_qa",
        subset_name=language,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        group_keys=["category"],  # Group by medical subject
    )


# Individual language wrapper functions
@task
def headqa_en(split: str = "test") -> Task:
    """HeadQA: English"""
    return headqa(language="en", split=split)


@task
def headqa_es(split: str = "test") -> Task:
    """HeadQA: Spanish"""
    return headqa(language="es", split=split)
