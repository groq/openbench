"""TriviaQA benchmark evaluation.

Large-scale reading comprehension dataset with trivia questions.
Evidence documents from Wikipedia and web search results.

Dataset: trivia_qa
Paper: TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.qa import qa_scorer

PROMPT_TEMPLATE = """Answer the following trivia question. Give a short, factual answer on the last line in the format "Answer: <your answer>".

Question: {question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a TriviaQA record to an Inspect Sample."""
    answer_data = record["answer"]
    # TriviaQA has multiple answer aliases
    aliases = answer_data.get("aliases", [])
    normalized = answer_data.get("normalized_aliases", [])
    value = answer_data.get("value", "")

    all_answers = set()
    if value:
        all_answers.add(value)
    all_answers.update(aliases)
    all_answers.update(normalized)

    target = "|".join(all_answers) if all_answers else ""

    return Sample(
        input=PROMPT_TEMPLATE.format(question=record["question"]),
        target=target,
    )


@task
def triviaqa(split: str = "validation") -> Task:
    """TriviaQA: Trivia question answering benchmark."""
    return Task(
        dataset=hf_dataset(
            path="trivia_qa",
            name="rc",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=qa_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=256),
    )
