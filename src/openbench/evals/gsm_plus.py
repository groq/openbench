"""GSM-Plus benchmark evaluation.

GSM-Plus is a dataset derived from GSM8K with perturbations to test robustness.
Includes numerical substitution, digit expansion, problem understanding changes.

Dataset: qintongli/GSM-Plus
Paper: GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.grade_school_math import grade_school_math_scorer

PROMPT_TEMPLATE = """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a GSM-Plus record to an Inspect Sample."""
    return Sample(
        input=PROMPT_TEMPLATE.format(question=record["question"]),
        target=str(record["answer"]),
        metadata={
            "answer_prefix": "Answer",
            "perturbation_type": record.get("perturbation_type", ""),
        },
    )


@task
def gsm_plus(split: str = "test") -> Task:
    """GSM-Plus: Robustness benchmark for grade school math."""
    return Task(
        dataset=hf_dataset(
            path="qintongli/GSM-Plus",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=grade_school_math_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=2048),
    )


@task
def gsm_plus_mini(split: str = "testmini") -> Task:
    """GSM-Plus Mini: Smaller test set (2400 samples)."""
    return Task(
        dataset=hf_dataset(
            path="qintongli/GSM-Plus",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[generate()],
        scorer=grade_school_math_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=2048),
    )
