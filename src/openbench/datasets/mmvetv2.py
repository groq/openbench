"""MM-Vet v2 dataset loader.

MM-Vet v2 is a challenging benchmark to evaluate large multimodal models for
integrated capabilities across recognition, OCR, knowledge, language generation,
spatial awareness, mathematics, and sequential reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, cast

from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

from openbench.utils.image import (
    compress_image,
    extract_image_bytes,
    image_bytes_to_data_uri,
)


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert an MM-Vet v2 record to an Inspect Sample.

    Args:
        record: Dataset record containing question, images, answer, capability

    Returns:
        Sample with multimodal input (text + images)
    """
    question = record["question"]
    answer = record["answer"]
    record_id = record["id"]
    capability = record.get("capability", [])
    added_in = record.get("added_in", "v2")

    # Build input content with text and images
    input_content: List[Union[ContentText, ContentImage]] = []

    # Process question text and image markers
    # Questions use format: "text<IMG><image_0>more text<IMG><image_1>..."
    # We need to interleave text and images in the correct order

    # Split by <IMG> marker and process
    parts = question.split("<IMG>")
    input_content.append(ContentText(text=parts[0]))  # Text before first image

    # Process each <image_N> marker and subsequent text
    for i, part in enumerate(parts[1:], start=0):
        # Extract image reference (e.g., "<image_0>") and remaining text
        if part.startswith("<image_"):
            # Find the end of the image marker
            end_idx = part.find(">")
            if end_idx != -1:
                img_marker = part[1:end_idx]  # Extract "image_0" from "<image_0>"
                remaining_text = part[end_idx + 1 :]  # Text after the marker

                # Get the image index
                img_idx = int(img_marker.split("_")[1])

                # Load and convert image if it exists
                image_data = record.get(f"image_{img_idx}")
                if image_data is not None:
                    try:
                        # Extract bytes from various image formats (HF dict, raw bytes, or PIL)
                        image_bytes = extract_image_bytes(image_data)

                        # Compress and convert to base64 data URI
                        compressed_bytes = compress_image(
                            image_bytes,
                            max_size_mb=5.0,
                            quality=75,
                            max_dimension=1536,
                        )
                        data_uri = image_bytes_to_data_uri(compressed_bytes)

                        # Add the image to input content
                        input_content.append(ContentImage(image=data_uri))
                    except ValueError:
                        # Skip if image format is unsupported
                        pass

                # Add any remaining text after this image marker
                if remaining_text.strip():
                    input_content.append(ContentText(text=remaining_text))

    metadata = {
        "question_id": record_id,
        "capability": capability,
        "added_in": added_in,
        "raw_question": question,
        "answer": answer,
    }

    return Sample(
        id=record_id,
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=answer,
        metadata=metadata,
    )


def get_mmvetv2_dataset() -> Dataset:
    """Load the MM-Vet v2 dataset.

    Returns:
        Dataset configured for MM-Vet v2 evaluation with 517 samples
    """
    dataset = hf_dataset(
        path="whyu/mm-vet-v2",
        split="test",
        sample_fields=record_to_sample,
    )

    samples = list(dataset)
    return MemoryDataset(samples=samples, name="mmvetv2")
