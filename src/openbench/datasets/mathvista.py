"""MathVista dataset loader.

Loads the MathVista dataset from HuggingFace and converts it to Inspect AI format.
Implements faithful reproduction of the original MathVista dataset processing.

Dataset: https://huggingface.co/datasets/AI4Math/MathVista
Paper: https://arxiv.org/abs/2310.02255
GitHub: https://github.com/lupantech/MathVista
"""

import io
from typing import Any, Callable, Dict, List, Optional, Union, cast

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from PIL import Image

from openbench.utils.image import (
    compress_image,
    extract_image_bytes,
    image_bytes_to_data_uri,
)


def record_to_sample(
    max_dimension: Optional[int] = 1536,
    quality: int = 75,
    max_size_mb: float = 5.0,
) -> Callable[[Dict[str, Any]], Sample]:
    """Creates a record-to-sample converter with specified image parameters.

    Args:
        max_dimension: Maximum width/height in pixels for image resizing.
                       If None, disables dimension-based resizing. (default: 1536)
        quality: JPEG quality (1-100) for image compression (default: 75)
        max_size_mb: Maximum allowed size in MB before compression (default: 5.0)

    Returns:
        A function that converts a MathVista record to an Inspect Sample
    """

    def _convert(record: Dict[str, Any]) -> Sample:
        """Convert a MathVista record to an Inspect Sample.

        Args:
            record: A record from the MathVista dataset

        Returns:
            An Inspect AI Sample with properly formatted input and metadata
        """
        # Extract core fields
        pid = str(record["pid"])
        question = record["question"]
        answer = record["answer"]
        question_type = record["question_type"]  # "multi_choice" or "free_form"
        answer_type = record["answer_type"]  # "text", "integer", or "float"
        query = record.get("query", "")  # Pre-formatted query with hints

        # Use pre-formatted query if available (faithful to original)
        prompt_text = query if query else question

        # Build input content with text first
        input_content: List[Union[ContentText, ContentImage]] = [
            ContentText(text=prompt_text)
        ]

        # Add the image if present
        if "decoded_image" in record and record["decoded_image"] is not None:
            image_data = record["decoded_image"]

            # Extract bytes from various image formats (HF dict, raw bytes, or PIL)
            image_bytes = extract_image_bytes(image_data)

            # Always process images through quality/dimension pipeline
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    # Convert to RGB if necessary (for JPEG compatibility)
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(
                            img,
                            mask=img.split()[-1]
                            if img.mode in ("RGBA", "LA")
                            else None,
                        )
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    # Resize if max_dimension is set and image exceeds it
                    if max_dimension is not None and max(img.size) > max_dimension:
                        img.thumbnail(
                            (max_dimension, max_dimension), Image.Resampling.LANCZOS
                        )

                    # Always re-encode at specified quality level
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=quality, optimize=True)
                    image_bytes = output.getvalue()
            except Exception:
                # If processing fails, use original bytes
                pass

            # Then apply additional size-based compression if needed
            compressed_bytes = compress_image(
                image_bytes,
                max_size_mb=max_size_mb,
                quality=quality,
                max_dimension=100000,  # Use very large value to skip dimension check
            )
            data_uri = image_bytes_to_data_uri(compressed_bytes)

            # Add the image to input content
            input_content.append(ContentImage(image=data_uri))

        # Extract metadata
        record_metadata = record.get("metadata", {})

        # Build comprehensive metadata (faithful to original structure)
        metadata = {
            "pid": pid,
            "question": question,
            "question_type": question_type,
            "answer_type": answer_type,
            "category": record.get("category", ""),
            "task": record.get("task", ""),
            "context": record.get("context", ""),
            "grade": record.get("grade", ""),
            "skills": record.get("skills", []),
            "unit": record.get("unit"),
            "precision": record.get("precision"),
            "choices": record.get("choices"),
        }

        # Add any additional metadata from the record
        if record_metadata:
            metadata["original_metadata"] = record_metadata

        return Sample(
            id=pid,
            input=[ChatMessageUser(content=cast(Any, input_content))],
            target=str(answer),
            metadata=metadata,
        )

    return _convert


def get_dataset(
    split: str = "testmini",
    question_type: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    max_dimension: Optional[int] = 1536,
    quality: int = 75,
    max_size_mb: float = 5.0,
) -> Dataset:
    """Load the MathVista dataset from HuggingFace.

    Args:
        split: Dataset split to load ("testmini" or "test")
        question_type: Optional filter by question type ("multi_choice" or "free_form")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        max_dimension: Maximum width/height in pixels for image resizing.
                       If None, disables dimension-based resizing. (default: 1536)
        quality: JPEG quality (1-100) for image compression (default: 75)
        max_size_mb: Maximum allowed size in MB before compression (default: 5.0)

    Returns:
        An Inspect AI Dataset
    """
    from inspect_ai.dataset import hf_dataset

    # Load from HuggingFace
    dataset = hf_dataset(
        path="AI4Math/MathVista",
        split=split,
        sample_fields=record_to_sample(
            max_dimension=max_dimension,
            quality=quality,
            max_size_mb=max_size_mb,
        ),
        shuffle=shuffle,
        seed=seed,
    )

    # Filter by question type if specified
    if question_type is not None:
        samples = [
            sample
            for sample in dataset
            if sample.metadata is not None
            and sample.metadata.get("question_type") == question_type
        ]
        dataset = MemoryDataset(samples=samples, name=f"mathvista_{split}")

    return dataset
