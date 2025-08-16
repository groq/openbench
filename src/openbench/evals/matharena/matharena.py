from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai import Task
from inspect_ai.model import GenerateConfig
from openbench.scorers import aime_scorer
from inspect_ai.solver import generate, prompt_template


def matharena_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["problem"],
        target=str(record["answer"]),
        id=record["problem_idx"],
        metadata={
            k: v
            for k, v in record.items()
            if k not in ["problem", "answer", "problem_idx"]
        },
    )


def matharena_task(
    dataset_path: str,
    instruction: str,
    name: str,
    default_max_tokens: int,
    default_temperature: float = 0.6,
    default_epochs: int = 4,
) -> Task:
    dataset = hf_dataset(
        path=dataset_path,
        split="train",
        sample_fields=matharena_record_to_sample,
    )

    TEMPLATE = instruction + "\n\n" + "{prompt}"
    return Task(
        dataset=dataset,
        solver=[prompt_template(TEMPLATE), generate()],
        scorer=aime_scorer(),  # Use specialized AIME scorer with robust extraction
        name=name,
        config=GenerateConfig(
            temperature=default_temperature,
            max_tokens=default_max_tokens,
        ),
        epochs=default_epochs,
    )
