"""MM-Vet v2 dataset loader.

MM-Vet v2 is a challenging benchmark to evaluate large multimodal models for
integrated capabilities across recognition, OCR, knowledge, language generation,
spatial awareness, mathematics, and sequential reasoning.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Union, cast

from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

from openbench.utils.image import compress_image, detect_image_mime_type


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
                    # Handle PIL Image objects from HuggingFace
                    if hasattr(image_data, "tobytes"):
                        # Convert PIL Image to bytes
                        import io

                        img_bytes = io.BytesIO()
                        image_data.save(img_bytes, format="PNG")
                        image_bytes = img_bytes.getvalue()
                    elif isinstance(image_data, dict) and "bytes" in image_data:
                        image_bytes = image_data["bytes"]
                    elif isinstance(image_data, bytes):
                        image_bytes = image_data
                    else:
                        image_bytes = None

                    if image_bytes:
                        # Compress and convert to base64 data URI
                        compressed_bytes = compress_image(
                            image_bytes,
                            max_size_mb=5.0,
                            quality=75,
                            max_dimension=1536,
                        )
                        base64_image = base64.b64encode(compressed_bytes).decode(
                            "utf-8"
                        )
                        mime_type = detect_image_mime_type(compressed_bytes)
                        data_uri = f"data:{mime_type};base64,{base64_image}"

                        # Add the image to input content
                        input_content.append(ContentImage(image=data_uri))

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
