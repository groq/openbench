"""DocVQA dataset loader.

Loads the DocVQA (Document Visual Question Answering) benchmark from HuggingFace
for evaluating models' ability to answer questions about document images.
Reference: https://arxiv.org/abs/2007.00398
"""

from typing import Any, Dict, List, Union, cast

from inspect_ai.dataset import Dataset, hf_dataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

from openbench.utils.image import (
    compress_image,
    extract_image_bytes,
    image_bytes_to_data_uri,
)


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a DocVQA record to an Inspect Sample.

    Args:
        record: DocVQA dataset record

    Returns:
        Sample with image and question as input, answer as target
    """
    question = record["question"]

    # Get answers
    answers = record.get("answers", [])
    if answers is None:
        answers = []

    # Target is the list of all acceptable answers
    target = answers

    # Build input with image and question
    input_content: List[Union[ContentText, ContentImage]] = []

    # Add image if present
    if "image" in record and record["image"] is not None:
        # Extract image bytes from HuggingFace dict format
        image_bytes = extract_image_bytes(record["image"])

        # Compress image to reduce size and speed up processing
        compressed_bytes = compress_image(
            image_bytes, max_size_mb=5.0, quality=85, max_dimension=2048
        )

        # Convert to base64 data URI
        data_uri = image_bytes_to_data_uri(compressed_bytes)

        input_content.append(ContentImage(image=data_uri))

    # Add question text
    input_content.append(ContentText(text=question))

    metadata = {
        "questionId": record["questionId"],
        "question": question,
        "docId": record["docId"],
        "ucsf_document_id": record["ucsf_document_id"],
        "ucsf_document_page_no": record["ucsf_document_page_no"],
        "answers": answers,
        "data_split": record["data_split"],
    }

    return Sample(
        id=str(record["questionId"]),
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=target,
        metadata=metadata,
    )


def get_docvqa_dataset(
    split: str,
) -> Dataset:
    """Load DocVQA dataset from HuggingFace.

    The DocVQA benchmark evaluates visual question answering on document images.
    The dataset contains questions about forms, reports, tables, diagrams, and
    other real-world documents requiring OCR and structural understanding.

    Args:
        split: Dataset split to load - "validation" (5,349 samples) or "test" (5,188 samples)

    Returns:
        Dataset with samples containing image + question as input, answer as target

    Note:
        - Validation split has ground truth answers for evaluation
        - Test split has no answers (for leaderboard submission only)
        - Images are compressed and converted in-memory (no disk caching)
    """
    # Validate split parameter
    if split not in ["validation", "test"]:
        raise ValueError(f"Invalid split '{split}'. Must be 'validation' or 'test'")

    # Use hf_dataset for lazy loading - images are processed on-demand
    return hf_dataset(
        path="lmms-lab/DocVQA",
        name="DocVQA",
        split=split,
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True,  # shuffle with fixed seed
        seed=42,
    )
