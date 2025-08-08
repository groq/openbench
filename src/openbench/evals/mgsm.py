"""MGSM (Multilingual Grade School Math) benchmark evaluation.

Based on: Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi et al., 2022
https://arxiv.org/abs/2210.03057
"""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mgsm import (
    get_dataset,
    LATIN_LANGUAGES,
    NON_LATIN_LANGUAGES,
)
from openbench.scorers.mgsm import mgsm_scorer


@task
def mgsm() -> Task:
    """MGSM: Multilingual Grade School Math - All languages.

    Evaluates mathematical reasoning across 11 languages including
    English, German, French, Spanish, Russian, Chinese, Japanese,
    Thai, Swahili, Bengali, and Telugu.

    Returns:
        Task configured for MGSM evaluation across all languages
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=mgsm_scorer(),
        name="mgsm",
        config=GenerateConfig(
            temperature=0.5,  # Simple Evals uses 0.5
            max_tokens=8192,  # Allow space for reasoning steps
        ),
    )


@task
def mgsm_en() -> Task:
    """MGSM: English only.

    Evaluates mathematical reasoning on English problems only.

    Returns:
        Task configured for MGSM evaluation in English
    """
    return Task(
        dataset=get_dataset(languages=["en"]),
        solver=[generate()],
        scorer=mgsm_scorer(),
        name="mgsm_en",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )


@task
def mgsm_latin() -> Task:
    """MGSM: Latin script languages.

    Evaluates mathematical reasoning across Latin script languages:
    German, English, Spanish, French, and Swahili.

    Returns:
        Task configured for MGSM evaluation on Latin script languages
    """
    return Task(
        dataset=get_dataset(languages=LATIN_LANGUAGES),
        solver=[generate()],
        scorer=mgsm_scorer(),
        name="mgsm_latin",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )


@task
def mgsm_non_latin() -> Task:
    """MGSM: Non-Latin script languages.

    Evaluates mathematical reasoning across non-Latin script languages:
    Bengali, Japanese, Russian, Telugu, Thai, and Chinese.

    Returns:
        Task configured for MGSM evaluation on non-Latin script languages
    """
    return Task(
        dataset=get_dataset(languages=NON_LATIN_LANGUAGES),
        solver=[generate()],
        scorer=mgsm_scorer(),
        name="mgsm_non_latin",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )
