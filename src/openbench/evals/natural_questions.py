"""Natural Questions Open benchmark evaluation.

Google's Natural Questions dataset in open-domain QA format.
Questions are real Google queries with answers from Wikipedia.

Dataset: google-research-datasets/nq_open
Paper: Natural Questions: A Benchmark for Question Answering Research
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.qa import qa_scorer

PROMPT_TEMPLATE = """Answer the following question. Give a short, factual answer on the last line in the format "Answer: <your answer>".

Question: {question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a Natural Questions record to an Inspect Sample."""
    answers = record["answer"]
    if isinstance(answers, list):
        target = "|".join(answers)
    else:
        target = str(answers)

    return Sample(
        input=PROMPT_TEMPLATE.format(question=record["question"]),
        target=target,
    )


@task
def natural_questions(split: str = "validation") -> Task:
    """Natural Questions Open: Open-domain QA benchmark."""
    return Task(
        dataset=hf_dataset(
            path="google-research-datasets/nq_open",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=qa_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=256),
    )
