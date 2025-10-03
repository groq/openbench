"""
XStoryCloze: Cross-lingual Story Cloze Test

XStoryCloze is a multilingual dataset for story completion and commonsense reasoning.
Given a four-sentence story, models must choose the correct ending from two options.
The dataset covers 11 languages to evaluate cross-lingual transfer capabilities.

Languages:
- en: English
- ru: Russian
- zh: Chinese
- es: Spanish
- ar: Arabic
- hi: Hindi
- id: Indonesian
- te: Telugu
- sw: Swahili
- eu: Basque
- my: Burmese

Sample usage:
```bash
bench eval xstorycloze_en --model "groq/llama-3.1-70b"
bench eval xstorycloze_zh --model "groq/llama-3.1-70b"
bench eval xstorycloze_ar --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{lin2022fewshot,
    title={Few-shot Learning with Multilingual Language Models},
    author={Xi Victoria Lin and Todor Mihaylov and Mikel Artetxe and Tianlu Wang and Shuohui Chen and Daniel Simig and Myle Ott and Naman Goyal and Shruti Bhosale and Jingfei Du and Ramakanth Pasunuru and Sam Shleifer and Punit Singh Koura and Vishrav Chaudhary and Brian O'Horo and Jeff Wang and Luke Zettlemoyer and Zornitsa Kozareva and Mona Diab and Veselin Stoyanov and Xian Li},
    booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
    year={2022}
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
XSTORYCLOZE_LANGUAGES = {
    "en": "English",
    "ru": "Russian",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "hi": "Hindi",
    "id": "Indonesian",
    "te": "Telugu",
    "sw": "Swahili",
    "eu": "Basque",
    "my": "Burmese",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert an XStoryCloze record to an OpenBench MCQSample.

    XStoryCloze tests story completion by asking which of two endings
    logically completes a four-sentence story.
    """
    # Build the story context from sentences 1-4
    story_context = "\n".join(
        [
            record["input_sentence_1"],
            record["input_sentence_2"],
            record["input_sentence_3"],
            record["input_sentence_4"],
        ]
    )

    input_question = f"{story_context}\n\nWhich sentence best completes the story?"

    # Two possible endings
    choice1 = record["sentence_quiz1"]
    choice2 = record["sentence_quiz2"]

    input_msg = prompt(
        question=input_question,
        choices=Choices([choice1, choice2]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # answer_right_ending is 1 or 2, convert to A or B
    answer = record["answer_right_ending"]
    int_to_char = {1: "A", 2: "B"}

    return MCQSample(
        input=input_msg,
        target=int_to_char[answer],
        metadata={
            "story_id": record.get("story_id"),
        },
    )


# Main XStoryCloze function - all language tasks call this
@task
def xstorycloze(language: str = "en", split: str = "eval") -> Task:
    """
    Family benchmark for XStoryCloze - run any language by code.

    Args:
        language: Language code to evaluate. Available languages:
                  - en: English (default)
                  - ru: Russian
                  - zh: Chinese
                  - es: Spanish
                  - ar: Arabic
                  - hi: Hindi
                  - id: Indonesian
                  - te: Telugu
                  - sw: Swahili
                  - eu: Basque
                  - my: Burmese
        split: Dataset split to use (default: "eval")
               Options: "train" (360 samples), "eval" (1510 samples)

    Returns:
        Task: The specified XStoryCloze language task

    Sample usage:
    ```bash
    # Run English XStoryCloze
    bench eval xstorycloze --model "openrouter/openai/gpt-oss-120b" -M only=groq

    # Programmatic access to specific language
    from openbench.evals.xstorycloze import xstorycloze
    task = xstorycloze(language="zh", split="eval")
    ```
    """
    if language not in XSTORYCLOZE_LANGUAGES:
        available = ", ".join(XSTORYCLOZE_LANGUAGES.keys())
        raise ValueError(
            f"Invalid XStoryCloze language '{language}'. Available: {available}"
        )

    return MCQEval(
        name=f"xstorycloze_{language}",
        dataset_path="juletxara/xstory_cloze",
        subset_name=language,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )


# Individual language wrapper functions
@task
def xstorycloze_en(split: str = "eval") -> Task:
    """XStoryCloze: English"""
    return xstorycloze(language="en", split=split)


@task
def xstorycloze_ru(split: str = "eval") -> Task:
    """XStoryCloze: Russian"""
    return xstorycloze(language="ru", split=split)


@task
def xstorycloze_zh(split: str = "eval") -> Task:
    """XStoryCloze: Chinese"""
    return xstorycloze(language="zh", split=split)


@task
def xstorycloze_es(split: str = "eval") -> Task:
    """XStoryCloze: Spanish"""
    return xstorycloze(language="es", split=split)


@task
def xstorycloze_ar(split: str = "eval") -> Task:
    """XStoryCloze: Arabic"""
    return xstorycloze(language="ar", split=split)


@task
def xstorycloze_hi(split: str = "eval") -> Task:
    """XStoryCloze: Hindi"""
    return xstorycloze(language="hi", split=split)


@task
def xstorycloze_id(split: str = "eval") -> Task:
    """XStoryCloze: Indonesian"""
    return xstorycloze(language="id", split=split)


@task
def xstorycloze_te(split: str = "eval") -> Task:
    """XStoryCloze: Telugu"""
    return xstorycloze(language="te", split=split)


@task
def xstorycloze_sw(split: str = "eval") -> Task:
    """XStoryCloze: Swahili"""
    return xstorycloze(language="sw", split=split)


@task
def xstorycloze_eu(split: str = "eval") -> Task:
    """XStoryCloze: Basque"""
    return xstorycloze(language="eu", split=split)


@task
def xstorycloze_my(split: str = "eval") -> Task:
    """XStoryCloze: Burmese"""
    return xstorycloze(language="my", split=split)
