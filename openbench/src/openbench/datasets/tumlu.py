from inspect_ai.dataset import Dataset, hf_dataset, Sample
from typing import Optional
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


def record_to_sample(record: dict[str, str]) -> Sample:
    return Sample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["question"],
            option_a=record["choices"][0],
            option_b=record["choices"][1],
            option_c=record["choices"][2],
            option_d=record["choices"][3],
        ),
        target=record["answer"],
        metadata={"subject": record["subject"]},
    )


def get_dataset(
    language: Optional[str] = "azerbaijani", shuffle: bool = False
) -> Dataset:
    return hf_dataset(
        path="jafarisbarov/TUMLU-mini",
        split="test",
        name=language,
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )
