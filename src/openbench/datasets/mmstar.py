from inspect_ai.dataset import Dataset, Sample, hf_dataset
import base64
from openbench.utils.image import detect_image_mime_type


def record_to_sample(record: dict) -> Sample:
    """Convert a MMStar record to an Inspect Sample."""

    meta_info = dict(record.get("meta_info", {}))
    meta_info["category"] = record.get("category")
    meta_info["subcategory"] = record.get("l2_category")
    meta_info["question"] = record.get("question")

    image_data = record.get("image")

    if isinstance(image_data, dict) and "bytes" in image_data:
        image_bytes = image_data["bytes"]
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        image_bytes = None

    if image_bytes:
        # Convert to base64 data URI with proper MIME type detection
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = detect_image_mime_type(image_bytes)
        data_uri = f"data:{mime_type};base64,{base64_image}"

        meta_info["image_uri"] = data_uri

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
