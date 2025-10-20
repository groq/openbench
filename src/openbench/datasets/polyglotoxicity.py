"""
PolygloToxicityPrompts Dataset Loader

Loads the PolygloToxicityPrompts dataset from HuggingFace.
This dataset measures neural toxic degeneration in LLMs across 17 languages.

Dataset: ToxicityPrompts/PolygloToxicityPrompts
Paper: https://arxiv.org/abs/2405.09373

The dataset contains naturally occurring prompts (~400 tokens average) across
17 languages with toxicity scores from Perspective API.
"""

from inspect_ai.dataset import Dataset, hf_dataset, Sample


# 17 languages supported by PolygloToxicityPrompts
POLYGLOTOXICITY_LANGUAGES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "zh": "Chinese",
}


def record_to_sample(record: dict) -> Sample:
    """Convert a PolygloToxicityPrompts record to an Inspect Sample.

    The dataset includes:
    - text: The prompt text (~400 tokens average)
    - toxicity: Original toxicity score from Perspective API
    - toxicity_bucket: Categorized toxicity level
    - prompt/continuation: Separated text components
    - perspective: Detailed Perspective API scores
    - meta_data: Contains language code in 'lang' field

    Args:
        record: Raw record from the HuggingFace dataset

    Returns:
        Sample with prompt as input and metadata for analysis
    """
    # Use 'prompt' field if available, otherwise use 'text'
    input_text = record.get("prompt", record.get("text", ""))

    # Extract language from meta_data field
    meta_data = record.get("meta_data", {})
    language = meta_data.get("lang", "en") if meta_data else "en"

    return Sample(
        input=input_text,
        target="",  # No target - we measure generated toxicity
        metadata={
            "language": language,
            "toxicity": record.get("toxicity", 0.0),
            "toxicity_bucket": record.get("toxicity_bucket", "unknown"),
            "original_text": record.get("text", ""),
        },
    )


def get_polyglotoxicity_dataset(
    language: str = "en",
    split: str = "full",
) -> Dataset:
    """Load the PolygloToxicityPrompts dataset for a specific language.

    Args:
        language: Language code (e.g., 'en', 'ar', 'de'). See POLYGLOTOXICITY_LANGUAGES
                 for all available languages.
        split: Dataset split - either "full" (complete dataset) or "small" (subset for testing)

    Returns:
        Dataset configured for the specified language and split

    Raises:
        ValueError: If language is not supported

    Examples:
        >>> # Load English dataset, full split
        >>> dataset = get_polyglotoxicity_dataset("en", "full")
        >>>
        >>> # Load Arabic dataset, small split for testing
        >>> dataset = get_polyglotoxicity_dataset("ar", "small")
    """
    if language not in POLYGLOTOXICITY_LANGUAGES:
        available = ", ".join(POLYGLOTOXICITY_LANGUAGES.keys())
        raise ValueError(
            f"Invalid language '{language}'. Available languages: {available}"
        )

    if split not in ["full", "small"]:
        raise ValueError(f"Invalid split '{split}'. Must be 'full' or 'small'")

    # HuggingFace config name format: ptp-{language_code}
    config_name = f"ptp-{language}"

    return hf_dataset(
        path="ToxicityPrompts/PolygloToxicityPrompts",
        name=config_name,
        split=split,
        sample_fields=record_to_sample,
        auto_id=True,
    )
