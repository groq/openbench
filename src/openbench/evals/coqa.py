"""CoQA benchmark evaluation.

Conversational Question Answering dataset with multi-turn dialogues.

Dataset: stanfordnlp/coqa
Paper: CoQA: A Conversational Question Answering Challenge
https://arxiv.org/abs/1808.07042
"""

from datasets import load_dataset as hf_load_dataset  # type: ignore[import-untyped]

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.qa import qa_scorer

PROMPT_TEMPLATE = """Read the story and answer the question. Give a short answer.

Story: {story}

Question: {question}

Answer:"""


def get_coqa_samples(split: str = "validation") -> list[Sample]:
    """Load and flatten CoQA dataset: one sample per question."""
    raw_dataset = hf_load_dataset(
        "stanfordnlp/coqa", split=split, trust_remote_code=True
    )
    samples = []

    for idx, record in enumerate(raw_dataset):
        story = record["story"]
        questions = record["questions"]
        answers = record["answers"]["input_text"]

        for q_idx, (question, answer) in enumerate(zip(questions, answers)):
            samples.append(
                Sample(
                    id=f"{idx}_{q_idx}",
                    input=PROMPT_TEMPLATE.format(story=story, question=question),
                    target=answer,
                    metadata={"source": record.get("source", "")},
                )
            )
    return samples


@task
def coqa(split: str = "validation") -> Task:
    """CoQA: Conversational question answering benchmark."""
    samples = get_coqa_samples(split)

    return Task(
        dataset=MemoryDataset(samples=samples, name="coqa"),
        solver=[generate()],
        scorer=qa_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=128),
    )
