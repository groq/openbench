from inspect_ai.dataset import Dataset, Sample, MemoryDataset, hf_dataset
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
from typing import List, Union, cast, Any
import base64
from openbench.utils.image import detect_image_mime_type


def record_to_sample(record: dict) -> Sample:
    """Convert an HLE record to an Inspect Sample."""
    # Format the input with the system prompt used in HLE
    input_text = record["question"]

    # Create multimodal content starting with the text
    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=input_text)
    ]

    # Include metadata for tracking
    metadata = {
        "question_id": record["id"],
    }

    # Handle multimodal questions by adding images to the input content
    if record.get("image"):
        image_data = record["image"]

        # Handle the image data - it should be in bytes format
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            # If it's a string, it might be base64 encoded or a URL
            # For now, store in metadata and skip adding to content
            metadata["image_url"] = str(image_data)
            image_bytes = None

        if image_bytes:
            # Convert to base64 data URI with proper MIME type detection
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = detect_image_mime_type(image_bytes)
            data_uri = f"data:{mime_type};base64,{base64_image}"

            # Add the image to the input content using data URI
            input_content.append(ContentImage(image=data_uri))
            metadata["image_url"] = data_uri

    return Sample(
        id=record["id"],
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=record["answer"],
        metadata=metadata,
    )


def get_dataset(text_only: bool = False) -> Dataset:
    """Load the HLE (Humanity's Last Exam) dataset.

    Args:
        text_only: If True, filter out multi-modal questions with images

    Returns:
        Dataset with HLE questions and answers
    """
    # Load the dataset from HuggingFace (no 'name' parameter - uses default config)
    dataset = hf_dataset(
        "cais/hle",
        split="test",
        sample_fields=record_to_sample,
    )

    # Convert to list for MemoryDataset
    samples = list(dataset)

    # Filter out image questions if text_only is True
    if text_only:
        samples = [
            s for s in samples if not (s.metadata and s.metadata.get("image_url"))
        ]
        dataset_name = "hle_text"
    else:
        dataset_name = "hle"

    return MemoryDataset(samples=samples, name=dataset_name)
