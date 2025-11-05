from inspect_ai.dataset import hf_dataset, Sample, MemoryDataset

AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=AIME_TEMPLATE.format(question=record["question"]),
        target=record["answer"],
    )


def get_dataset() -> MemoryDataset:
    # Load both files separately and combine them
    all_samples = []

    # Load aime2025-I.jsonl
    dataset_i = hf_dataset(
        path="opencompass/AIME2025",
        split="test",
        name="AIME2025-I",
        sample_fields=record_to_sample,
    )
    all_samples.extend(list[Sample](dataset_i))

    # Load aime2025-II.jsonl
    dataset_ii = hf_dataset(
        path="opencompass/AIME2025",
        split="test",
        name="AIME2025-II",
        sample_fields=record_to_sample,
    )
    all_samples.extend(list[Sample](dataset_ii))

    return MemoryDataset(samples=all_samples, name="gpt_oss_aime2025")
