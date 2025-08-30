from inspect_ai.dataset import Dataset, hf_dataset, Sample
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
from typing import Dict, Any, List, Optional, Union, cast
import base64
import string
from openbench.utils.image import detect_image_mime_type


# Need for dynamic prompt building b/c MMMU Pro has multiple choice questions with many options
def _build_dynamic_mcq_prompt(prompt: str, options: List[str]) -> str:
    letters = string.ascii_uppercase[: len(options)]
    letters_str = "".join(letters)
    lines = [
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {}.".format(
            letters_str
        ),
        "",
        prompt,
        "",
    ]
    for idx, text in enumerate(options):
        lines.append(f"{letters[idx]}) {text}")
    return "\n".join(lines).strip()


def _extract_option_texts(options_obj: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(options_obj, list):
        for opt in options_obj:
            if isinstance(opt, dict):
                texts.append(str(opt.get("text", opt.get("value", "")).strip()))
            else:
                texts.append(str(opt).strip())
    else:
        texts.append(str(options_obj).strip())
    return texts


def record_to_sample(record: Dict[str, Any]) -> Sample:
    # Question text may be missing for vision-only samples
    question_raw = record.get("question")
    question = (
        str(question_raw).strip()
        if question_raw
        else "Use the image to answer the question. Choose the best option."
    )

    options = record.get("options", [])
    answer = record.get("answer", "")

    # Normalize options
    option_texts = _extract_option_texts(options) if options else []

    # Build prompt
    full_question = (
        _build_dynamic_mcq_prompt(question, option_texts) if option_texts else question
    )

    input_content: List[Union[ContentText, ContentImage]] = [
        ContentText(text=full_question)
    ]

    num_images = 0

    # Single image field
    if "image" in record and record["image"] is not None:
        image_val = record["image"]
        image_bytes: Optional[bytes] = None
        if isinstance(image_val, dict) and "bytes" in image_val:
            image_bytes = image_val["bytes"]
        elif isinstance(image_val, (bytes, bytearray)):
            image_bytes = bytes(image_val)
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = detect_image_mime_type(image_bytes)
            data_uri = f"data:{mime_type};base64,{base64_image}"
            input_content.append(ContentImage(image=data_uri))
            num_images += 1

    # Multiple images
    for i in range(1, 8):
        image_key = f"image_{i}"
        if image_key in record and record[image_key] is not None:
            image_data = record[image_key]
            try:
                image_bytes2 = image_data["bytes"]
            except Exception:
                continue
            base64_image2 = base64.b64encode(image_bytes2).decode("utf-8")
            mime_type2 = detect_image_mime_type(image_bytes2)
            data_uri2 = f"data:{mime_type2};base64,{base64_image2}"
            input_content.append(ContentImage(image=data_uri2))
            num_images += 1

    metadata = {
        "question_id": record.get("id", ""),
        "options": options,
        "num_images": num_images,
        "question": question,
        "answer": answer,
        "subfield": record.get("subfield", ""),
        "topic_difficulty": record.get("topic_difficulty", ""),
        "category": record.get("category", record.get("subject", "")),
    }

    return Sample(
        id=str(record.get("id", "")),
        input=[ChatMessageUser(content=cast(Any, input_content))],
        target=str(answer),
        metadata=metadata,
    )


def get_dataset(subset: Optional[str] = None) -> Dataset:
    return hf_dataset(
        path="MMMU/MMMU_Pro",
        split="test",
        name=subset,
        sample_fields=record_to_sample,
    )
