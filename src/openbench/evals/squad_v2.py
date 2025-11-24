"""SQuAD v2 benchmark evaluation.

Stanford Question Answering Dataset 2.0 with unanswerable questions.
Reading comprehension over Wikipedia passages.

Dataset: squad_v2
Paper: Know What You Don't Know: Unanswerable Questions for SQuAD
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.qa import qa_scorer

PROMPT_TEMPLATE = """Read the passage and answer the question. If the question cannot be answered from the passage, respond with "unanswerable". Give your answer on the last line in the format "Answer: <your answer>".

Passage: {context}

Question: {question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a SQuAD v2 record to an Inspect Sample."""
    answers = record["answers"]["text"]
    if not answers:
        # Unanswerable question
        target = "unanswerable"
    else:
        # Use unique answers
        target = "|".join(set(answers))

    return Sample(
        input=PROMPT_TEMPLATE.format(
            context=record["context"],
            question=record["question"],
        ),
        target=target,
        metadata={"title": record.get("title", "")},
    )


@task
def squad_v2(split: str = "validation") -> Task:
    """SQuAD v2: Reading comprehension with unanswerable questions."""
    return Task(
        dataset=hf_dataset(
            path="squad_v2",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=qa_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=256),
    )
