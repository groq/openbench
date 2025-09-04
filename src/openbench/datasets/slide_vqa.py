from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
from typing import List, Union, cast, Any
import base64
from openbench.utils.image import detect_image_mime_type


def record_to_sample(record: dict) -> Sample:
    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=record["question"])
    ]

    evidence_pages = record.get("evidence_pages", [])

    for page_num in range(1, 21):
        page_key = f"page_{page_num}"
        if page_key in record and record[page_key] is not None:
            if not evidence_pages or page_num in evidence_pages:
                image_data = record[page_key]

                if isinstance(image_data, dict) and "bytes" in image_data:
                    image_bytes = image_data["bytes"]
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                else:
                    continue

                if image_bytes:
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    mime_type = detect_image_mime_type(image_bytes)
                    data_uri = f"data:{mime_type};base64,{base64_image}"
                    input_content.append(ContentImage(image=data_uri))

    return Sample(
        id=str(record["qa_id"]),
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=record["answer"],
        metadata={
            "qa_id": record["qa_id"],
            "deck_name": record.get("deck_name"),
            "deck_url": record.get("deck_url"),
            "arithmetic_expression": record.get("arithmetic_expression"),
            "evidence_pages": evidence_pages,
        },
    )


def get_dataset(split: str = "test") -> Dataset:
    """Load the SlideVQA dataset.

    Args:
        split: Dataset split to load ('train', 'test', or 'eval')

    Returns:
        Dataset with SlideVQA questions and answers
    """
    return hf_dataset(
        path="NTT-hil-insight/SlideVQA",
        split=split,
        sample_fields=record_to_sample,
    )
