"""MMMU (Massive Multi-discipline Multimodal Understanding) dataset loader."""

from inspect_ai.dataset import Dataset, hf_dataset, Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
from typing import Dict, Any, List, Optional, Union, cast
import base64
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE
from openbench.utils.image import detect_image_mime_type


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert an MMMU record to an Inspect Sample."""

    question = record["question"]
    options = record["options"]
    record_id = record["id"]
    answer = record["answer"]

    # Handle different possible formats for options
    if isinstance(options, str):
        if options.strip().startswith("[") and options.strip().endswith("]"):
            try:
                import ast

                parsed_options = ast.literal_eval(options)
                if isinstance(parsed_options, list):
                    options = parsed_options
            except (ValueError, SyntaxError):
                pass

    # Extract individual options
    if isinstance(options, list):
        # Ensure we have exactly 4 options (pad with empty if needed)
        while len(options) < 4:
            options.append("")

        # Get option text
        option_texts = []
        for option in options[:4]:  # Only use first 4 options
            if isinstance(option, dict):
                option_texts.append(option.get("text", str(option)))
            else:
                option_texts.append(str(option))

        # Use the standard multiple choice template
        full_question = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=question,
            option_a=option_texts[0],
            option_b=option_texts[1],
            option_c=option_texts[2],
            option_d=option_texts[3],
        )
    else:
        # Fallback if options aren't in expected format
        full_question = f"{question}\n\nOptions: {options}\n\nAnswer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."

    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=full_question)
    ]

    # Handle Multimodal Questions by adding images to the input content via ContentImage
    num_images = 0
    for i in range(1, 8):
        image_key = f"image_{i}"
        if image_key in record and record[image_key] is not None:
            image_data = record[image_key]

            image_bytes = image_data["bytes"]

            # Convert to base64 data URI with proper MIME type detection
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = detect_image_mime_type(image_bytes)
            data_uri = f"data:{mime_type};base64,{base64_image}"

            # Add the image to the input content using data URI
            input_content.append(ContentImage(image=data_uri))
            num_images += 1

    metadata = {
        "question_id": record_id,
        "options": options,
        "img_type": record.get("img_type", []),
        "topic_difficulty": record.get("topic_difficulty", ""),
        "question_type": record.get("question_type", "multiple-choice"),
        "subfield": record.get("subfield", ""),
        "explanation": record.get("explanation", ""),
        "num_images": num_images,
    }

    return Sample(
        id=record_id,
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=answer,
        metadata=metadata,
    )


def get_dataset(
    subset: Optional[str] = None,
    split: str = "validation",
) -> Dataset:
    """Load the MMMU dataset.

    Args:
        subset: Optional subset name (e.g., "Accounting", "Art", "Biology", etc.)
               If None, loads all subsets combined
        split: Dataset split to use ("dev", "validation", "test")
        num_examples: Optional limit on number of examples to load

    Returns:
        Dataset configured for MMMU evaluation
    """
    if subset:
        # Load specific subset
        dataset = hf_dataset(
            path="MMMU/MMMU",
            name=subset,
            split=split,
            sample_fields=record_to_sample,
        )
        samples = list(dataset)
        dataset_name = f"mmmu_{subset.lower()}"
    else:
        # Load all subsets and combine them
        all_samples = []
        available_subsets = get_available_subsets()

        for subset_name in available_subsets:
            subset_dataset = hf_dataset(
                path="MMMU/MMMU",
                name=subset_name,
                split=split,
                sample_fields=record_to_sample,
            )
            subset_samples = list(subset_dataset)
            all_samples.extend(subset_samples)

        samples = all_samples
        dataset_name = "mmmu"

    return MemoryDataset(samples=samples, name=dataset_name)


def get_available_subsets() -> List[str]:
    """Get list of available MMMU subsets/subjects."""
    return [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]
