"""ChartQAPro dataset loader."""

from __future__ import annotations

from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

from openbench.utils.image import extract_image_bytes, image_bytes_to_data_uri


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    Convert a ChartQAPro record to an Inspect Sample.

    The dataset contains:
    - image: Embedded JPEG bytes
    - Question: Array of 1-7 questions
    - Answer: Array of 1-7 answers
    - Question_Type: Category (Factoid, Multi Choice, Hypothetical, Fact Checking, Conversational)
    - Year: Array of year flags for scoring (YES/NO)
    - Paragraph: Optional context text

    For Conversational questions: Only the LAST answer is scored (per official evaluation).
    For other types: Typically single question, but can have multiple.

    Args:
        record: Dictionary from HuggingFace dataset

    Returns:
        Sample with image, first question, and all metadata for solver/scorer
    """
    # Extract and convert image (stored as raw bytes in Parquet)
    image_bytes = extract_image_bytes(record["image"])

    # Convert to base64 data URI with proper MIME type detection
    data_uri = image_bytes_to_data_uri(image_bytes)

    # Extract all fields
    questions = record["Question"]  # List of 1-7 questions
    answers = record["Answer"]  # List of 1-7 answers
    question_type = record["Question Type"]  # Note: has SPACE not underscore
    year_flags = record.get("Year", [])  # Year flags for scoring logic
    paragraph = record.get("Paragraph", "")  # Optional context

    # Create initial input with image + first question
    # Solver will handle multi-turn conversation if needed
    image_content = ContentImage(image=data_uri)
    text_content = ContentText(text=questions[0])

    # For Conversational questions: only last answer is scored per official eval
    # For other types: use all answers (though typically just 1)
    target_answers = answers if question_type != "Conversational" else [answers[-1]]

    return Sample(
        id=str(record.get("id", hash(str(record)))),
        input=[ChatMessageUser(content=[image_content, text_content])],
        target=target_answers[-1],  # Single answer for scoring
        metadata={
            "image_data_uri": data_uri,
            "questions": questions,
            "answers": answers,
            "question_type": question_type,
            "year_flags": year_flags,
            "paragraph": paragraph,
            "num_questions": len(questions),
        },
    )


def get_chartqapro_dataset() -> Dataset:
    """
    Load the ChartQAPro dataset from HuggingFace.

    Dataset: ahmed-masry/ChartQAPro
    Split: test (1,950 samples)

    Returns 1,950 samples covering:
    - 5 question types: Factoid, Multi Choice, Hypothetical, Fact Checking, Conversational
    - 1-7 questions per chart
    - 157 diverse sources
    - Images with optional context paragraphs

    Returns:
        MemoryDataset with all samples loaded
    """
    dataset = hf_dataset(
        path="ahmed-masry/ChartQAPro",
        split="test",
        sample_fields=record_to_sample,
    )

    samples = list(dataset)
    return MemoryDataset(samples=samples, name="chartqapro")
