"""OCRBench v2 dataset loader.

Dataset: https://huggingface.co/datasets/morpheushoc/OCRBenchv2
Paper: https://arxiv.org/abs/2501.00321
"""

from __future__ import annotations

from datasets import load_dataset as hf_load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import MemoryDataset, Sample

from openbench.utils.image import image_bytes_to_data_uri, pil_image_to_bytes

# Import capability mapping from scorer
# We need to add capability to metadata at dataset level for grouped metrics
from openbench.scorers.ocrbenchv2 import CAPABILITY_MAPPING


# Chinese categories for language filtering
CN_CATEGORIES = {
    "cognition VQA cn",
    "key information extraction cn",
    "formula recognition cn",
    "full-page OCR cn",
    "reasoning VQA cn",
    "text translation cn",
    "table parsing cn",
    "handwritten answer extraction cn",
    "document parsing cn",
}

# All valid categories from the dataset
ALL_CATEGORIES = {
    # English
    "APP agent en",
    "ASCII art classification en",
    "key information extraction en",
    "key information mapping en",
    "math QA en",
    "full-page OCR en",
    "reasoning VQA en",
    "fine-grained text recognition en",
    "science QA en",
    "table parsing en",
    "text counting en",
    "text grounding en",
    "text recognition en",
    "text spotting en",
    "document classification en",
    "cognition VQA en",
    "VQA with position en",
    "chart parsing en",
    "document parsing en",
    "formula recognition en",
    "diagram QA en",
    # Chinese
    "cognition VQA cn",
    "key information extraction cn",
    "formula recognition cn",
    "full-page OCR cn",
    "reasoning VQA cn",
    "text translation cn",
    "table parsing cn",
    "handwritten answer extraction cn",
    "document parsing cn",
}


def get_ocrbenchv2_dataset(
    categories: list[str] | None = None,
    language: str = "all",
    limit: int | None = None,
    seed: int | None = None,
) -> MemoryDataset:
    """
    Load OCRBench v2 dataset from Hugging Face.

    Args:
        categories: List of specific categories to include (e.g., ["APP agent en", "table parsing en"])
        language: Filter by language - "en" (English), "cn" (Chinese), or "all" (default)
        limit: Maximum number of samples to load
        seed: Random seed for shuffling (to get diverse samples across categories)

    Returns:
        MemoryDataset with samples containing image + question
    """
    # Load from HuggingFace
    hf_dataset = hf_load_dataset(
        "morpheushoc/OCRBenchv2", split="test", trust_remote_code=True
    )

    # Shuffle if seed provided (helps get diverse categories when using limit)
    if seed is not None:
        hf_dataset = hf_dataset.shuffle(seed=seed)

    samples = []
    for item in hf_dataset:
        dataset_name = item["dataset_name"]
        category = item.get("type", "")

        # Apply category filter
        if categories and category not in categories:
            continue

        # Apply language filter
        if language == "en" and category in CN_CATEGORIES:
            continue
        elif language == "cn" and category not in CN_CATEGORIES:
            continue

        # Convert PIL Image to base64 data URI
        pil_image = item.get("image")
        if pil_image is None:
            continue  # Skip samples without images

        # Convert to RGB if necessary (JPEG doesn't support RGBA)
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")

        # Convert PIL Image to bytes and create data URI
        image_bytes = pil_image_to_bytes(pil_image, format="JPEG")
        image_uri = image_bytes_to_data_uri(image_bytes)

        # Target is list of acceptable answers
        target = item["answers"] if item["answers"] else [""]

        # Map category to capability
        capability = CAPABILITY_MAPPING.get(category, "Unknown")

        samples.append(
            Sample(
                input=item["question"],
                target=target,
                id=f"{dataset_name}_{item['id']}",
                metadata={
                    "dataset_name": dataset_name,
                    "type": category,
                    "capability": capability,  # Add capability for grouped metrics
                    "image_uri": image_uri,
                    "bbox": item.get("bbox"),
                    "raw_text": item.get("raw_text"),
                    "content": item.get("content"),
                    "image_path": item.get("image_path", ""),
                    "question": item[
                        "question"
                    ],  # Add question for handwritten CN essay detection
                },
            )
        )

        # Apply limit
        if limit and len(samples) >= limit:
            break

    return MemoryDataset(samples=samples, name="OCRBench v2")
