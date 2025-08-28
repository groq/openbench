"""MMMU (Massive Multi-discipline Multimodal Understanding) dataset loader."""

from inspect_ai.dataset import Dataset, hf_dataset, Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
from typing import Dict, Any, List, Optional, Union, cast
from PIL import Image
import os
import io
import tempfile
import atexit
import shutil

# Global temp directory for MMMU images
TEMP_IMAGE_DIR = tempfile.mkdtemp()


def _cleanup_temp_images():
    """Clean up temporary images directory when process exits."""
    try:
        if os.path.exists(TEMP_IMAGE_DIR):
            shutil.rmtree(TEMP_IMAGE_DIR)
    except Exception:
        # Silently ignore cleanup failures to avoid interfering with process exit
        pass


# Register cleanup function to run when process exits
atexit.register(_cleanup_temp_images)


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
                else:
                    option_text = options
            except (ValueError, SyntaxError):
                option_text = options
        else:
            option_text = options

    if isinstance(options, list):
        # Standard case: options is a list of strings
        option_text = ""
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            if isinstance(option, dict):
                option_str = option.get("text", str(option))
            else:
                option_str = str(option)
            option_text += f"{letter}. {option_str}\n"
    elif not isinstance(options, str):
        # Fallback: convert to string
        option_text = str(options)

    # Add zero-shot chain of though reasoning to task prompt
    chain_of_thought_reasoning = "The final answer is: "

    full_question = f"{question}\n\n{option_text}\n\n{chain_of_thought_reasoning}"

    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=full_question)
    ]

    image_paths = []
    # Handle Multimodal Questions by adding images to the input content via ContentImage
    for i in range(1, 8):
        image_key = f"image_{i}"
        if record.get(image_key) and record[image_key].get("bytes"):
            image_bytes = record[image_key]["bytes"]

            # Load bytes into a PIL Image object to validate it's a valid image
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Create a unique path in our session's temp directory
            temp_file_path = os.path.join(TEMP_IMAGE_DIR, f"{record_id}_img_{i}.png")

            # Save the image and get its path
            pil_image.save(temp_file_path)
            image_paths.append(temp_file_path)

            # Add the image to the input content
            input_content.append(ContentImage(image=temp_file_path))

    metadata = {
        "question_id": record_id,
        "options": options,
        "img_type": record.get("img_type", []),
        "topic_difficulty": record.get("topic_difficulty", ""),
        "question_type": record.get("question_type", "multiple-choice"),
        "subfield": record.get("subfield", ""),
        "explanation": record.get("explanation", ""),
        "image_paths": image_paths,
        "num_images": len(image_paths),
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
    num_examples: Optional[int] = None,
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
            try:
                subset_dataset = hf_dataset(
                    path="MMMU/MMMU",
                    name=subset_name,
                    split=split,
                    sample_fields=record_to_sample,
                )
                subset_samples = list(subset_dataset)
                all_samples.extend(subset_samples)
            except Exception:
                continue

        samples = all_samples
        dataset_name = "mmmu"

    # Limit number of examples if specified
    if num_examples is not None:
        samples = samples[:num_examples]

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
