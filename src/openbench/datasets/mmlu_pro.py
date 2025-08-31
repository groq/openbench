from inspect_ai.dataset import Dataset, hf_dataset, Sample


def record_to_sample(record: dict[str, str]) -> Sample:
    prompt_list = [
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD etc.",
        "",
        f"{record['question']}",
        "",
    ]

    for i, option in enumerate(record["options"]):
        letter = chr(ord("A") + i)
        prompt_list.append(f"{letter}) {option}")

    prompt_str = "\n".join(prompt_list)

    return Sample(
        input=prompt_str,
        target=record["answer"],
        metadata={
            "category": record["category"],
            "src": record["src"],
        },
    )


def get_dataset(shuffle: bool = False) -> Dataset:
    """
    Load the MMLU Pro dataset for evaluation.

    Returns:
        Dataset: MMLU-Pro dataset.
    """
    return hf_dataset(
        path="TIGER-Lab/MMLU-Pro",
        split="test",
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )
