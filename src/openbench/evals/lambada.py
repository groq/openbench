"""LAMBADA benchmark evaluation.

Language Modeling Broadened to Account for Discourse Aspects.
Tests ability to predict the last word of a passage requiring
broad context understanding.

Dataset: lambada
Paper: The LAMBADA Dataset (Paperno et al., 2016)
https://arxiv.org/abs/1606.06031
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate

PROMPT_TEMPLATE = """Complete the following passage with the final word. Output only the single word that completes the passage.

{context}

The final word is:"""


def record_to_sample(record: dict) -> Sample:
    """Convert a LAMBADA record to an Inspect Sample."""
    text = record["text"]
    # Split into context and target (last word)
    words = text.split()
    target = words[-1].rstrip(".,!?;:'\"")
    context = " ".join(words[:-1])

    return Sample(
        input=PROMPT_TEMPLATE.format(context=context),
        target=target,
    )


@task
def lambada(split: str = "test") -> Task:
    """LAMBADA: Language modeling with broad discourse context."""
    return Task(
        dataset=hf_dataset(
            path="lambada",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=match(ignore_case=True),
        config=GenerateConfig(temperature=0.0, max_tokens=16),
    )
