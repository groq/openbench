from inspect_ai.dataset import Dataset, Sample, hf_dataset

from openbench.utils.image import extract_image_bytes, image_bytes_to_data_uri


def record_to_sample(record: dict) -> Sample:
    """Convert a MMStar record to an Inspect Sample."""

    meta_info = dict(record.get("meta_info", {}))
    meta_info["category"] = record.get("category")
    meta_info["subcategory"] = record.get("l2_category")
    meta_info["question"] = record.get("question")

    image_data = record.get("image")

    if image_data:
        try:
            # Extract bytes from various image formats (HF dict, raw bytes, or PIL)
            image_bytes = extract_image_bytes(image_data)
            # Convert to base64 data URI with proper MIME type detection
            data_uri = image_bytes_to_data_uri(image_bytes)
            meta_info["image_uri"] = data_uri
        except ValueError:
            # Image data format not supported, skip image
            pass

    return Sample(
        id=record.get("index"),
        input="",
        target=record.get("answer", ""),
        metadata=meta_info,
    )


def get_mmstar_dataset() -> Dataset:
    return hf_dataset(
        path="Lin-Chen/MMStar",
        split="val",  # only validation split is available
        sample_fields=record_to_sample,
    )
