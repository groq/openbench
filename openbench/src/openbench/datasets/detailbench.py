from inspect_ai.dataset import Dataset, Sample, MemoryDataset, hf_dataset


def record_to_sample(record: dict) -> Sample:
    """Convert a DetailBench record to an Inspect Sample.

    Expected fields from the HF dataset:
    - article_title: str
    - original_text: str
    - modified_text: str
    - original_number: str/int
    - modified_number: str/int
    - change_position: int (index of modified number in modified_text)
    - target_language: str (language to translate into)
    """

    # Prompt to eval
    target_language = record.get("target_language", "")
    modified_text = record.get("modified_text", "")
    translate_prompt = (
        f"Translate the following text to {target_language}:\n\n{modified_text}"
    )

    # Preserve raw fields for the scorer (judge)
    metadata = {
        "original_text": record.get("original_text", ""),
        "modified_text": modified_text,
        "original_number": str(record.get("original_number", "")),
        "modified_number": str(record.get("modified_number", "")),
        "change_position": int(record.get("change_position", 0) or 0),
        "target_language": target_language,
    }

    # No strict gold target for generation; grading is model-based
    return Sample(input=translate_prompt, target="", metadata=metadata)


def get_dataset() -> Dataset:
    dataset = hf_dataset(
        "xeophon/detailbench",
        split="train",
        sample_fields=record_to_sample,
    )

    samples = list(dataset)
    return MemoryDataset(samples=samples, name="detailbench")
