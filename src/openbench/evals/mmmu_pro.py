from typing import List, Union, Dict, Any, Optional, cast
import ast
from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig, ChatMessageUser, ContentText, ContentImage
from openbench.utils.mcq import MCQSample, MCQEval
from openbench.utils.text import create_dynamic_multiple_choice_prompt
from openbench.utils.image import (
    compress_image,
    extract_image_bytes,
    image_bytes_to_data_uri,
)


def _parse_options_string(options_string: str) -> List[str]:
    """Parse options from string representation of a list."""
    try:
        # Use ast.literal_eval for safe evaluation of string representations
        parsed_list = ast.literal_eval(options_string.strip())
        return [str(option).strip() for option in parsed_list]
    except (ValueError, SyntaxError):
        # If parsing fails, return empty list
        return []


def record_to_mcq_sample(record: Dict[str, Any]) -> MCQSample:
    """Convert a MMMU Pro record to an openbench MCQSample."""
    # Question text may be missing for vision-only samples
    question_raw = record.get("question")
    question = (
        str(question_raw).strip()
        if question_raw
        else "Use the image to answer the question. Choose the best option."
    )

    options_string = record.get("options", "")
    answer = record.get("answer", "")

    # Parse options from string format
    options = _parse_options_string(options_string)

    # Build prompt dynamically based on available options
    full_question = create_dynamic_multiple_choice_prompt(question, options)

    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=full_question)
    ]

    num_images = 0

    # Single image field
    if "image" in record and record["image"] is not None:
        image_val = record["image"]
        try:
            # Extract bytes from various image formats (HF dict, raw bytes, or PIL)
            image_bytes = extract_image_bytes(image_val)
            compressed_bytes = compress_image(
                image_bytes, max_size_mb=5.0, quality=75, max_dimension=1536
            )
            data_uri = image_bytes_to_data_uri(compressed_bytes)
            input_content.append(ContentImage(image=data_uri))
            num_images += 1
        except ValueError:
            # Skip if image format is unsupported
            pass

    # Multiple images
    for i in range(1, 8):
        image_key = f"image_{i}"
        if image_key in record and record[image_key] is not None:
            image_data = record[image_key]
            try:
                # Extract bytes from various image formats (HF dict, raw bytes, or PIL)
                image_bytes = extract_image_bytes(image_data)
                # Compress image if too large to avoid 413 errors
                compressed_bytes = compress_image(
                    image_bytes, max_size_mb=5.0, quality=75, max_dimension=1536
                )
                data_uri = image_bytes_to_data_uri(compressed_bytes)
                input_content.append(ContentImage(image=data_uri))
                num_images += 1
            except (ValueError, Exception):
                # Skip if image format is unsupported or extraction fails
                continue

    metadata = {
        "question_id": record.get("id", ""),
        "options": options,  # Parsed list of options
        "options_string": options_string,  # Original string for reference
        "num_images": num_images,
        "question": question,
        "answer": answer,
        "subfield": record.get("subfield", ""),
        "topic_difficulty": record.get("topic_difficulty", ""),
        "category": record.get("category", record.get("subject", "")),
    }

    return MCQSample(
        id=str(record.get("id", "")),
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=str(answer),
        metadata=metadata,
    )


@task
def mmmu_pro(subset: Optional[str] = "standard (10 options)") -> Task:
    """
    MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding
    capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU
    benchmark by introducing several key improvements that make it more challenging and realistic,
    ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual
    and textual information.
    """
    return MCQEval(
        name="mmmu_pro",
        dataset_path="MMMU/MMMU_Pro",
        split="test",
        subset_name=subset,
        record_to_mcq_sample=record_to_mcq_sample,
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )


@task
def mmmu_pro_vision() -> Task:
    return mmmu_pro(subset="vision")
