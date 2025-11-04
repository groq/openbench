import uuid
from typing import Any, Dict, List
from inspect_ai.dataset import hf_dataset, Sample, Dataset
from inspect_ai.model import ContentImage, ContentText, ChatMessageUser
from openbench.utils.image import detect_image_mime_type
import base64
import io
from PIL import Image
from io import BytesIO


def preprocess_image(input):
    """Preprocess image to data URI format with resizing to 1024x1024"""
    image_bytes = input["bytes"]

    # Resize the image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((1024, 1024))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Create proper data URI with MIME type detection
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = detect_image_mime_type(img_bytes)
    data_uri = f"data:{mime_type};base64,{base64_image}"

    return data_uri


def record_to_sample_rocketscience(record: Dict[str, Any]) -> List[Sample]:
    """Convert a RocketScience record to 4 Inspect Samples following original methodology.

    RocketScience creates 4 evaluations per dataset item:
    1. Select best text for image1 (expected: "1")
    2. Select best text for image2 (expected: "2")
    3. Select best image for text1 (expected: "1")
    4. Select best image for text2 (expected: "2")
    """

    tuple_id = uuid.uuid4()  # Unique ID for the group of 4 samples
    text1 = record["text1"]
    text2 = record["text2"]
    image1 = preprocess_image(record["image1"])
    image2 = preprocess_image(record["image2"])

    select_text_prompt = f'Which caption fits the image best? Reason about it and at the end write "RESPONSE" and reply only with the number 1 or 2. 1.) {text1} 2.) {text2}'
    select_image_prompt1 = f'Which image fits the caption best? Reason about it and at the end write "RESPONSE" and reply only with the number 1 or 2. Caption: {text1}'
    select_image_prompt2 = f'Which image fits the caption best? Reason about it and at the end write "RESPONSE" and reply only with the number 1 or 2. Caption: {text2}'

    return [
        Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentText(text=select_text_prompt),
                        ContentImage(image=image1),
                    ]
                )
            ],
            target=["1"],
            metadata={"tuple_id": tuple_id, "type": "textscore"},
        ),
        Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentText(text=select_text_prompt),
                        ContentImage(image=image2),
                    ]
                )
            ],
            target=["2"],
            metadata={"tuple_id": tuple_id, "type": "textscore"},
        ),
        Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentText(text=select_image_prompt1),
                        ContentImage(image=image1),
                        ContentImage(image=image2),
                    ]
                )
            ],
            target=["1"],
            metadata={"tuple_id": tuple_id, "type": "imagescore"},
        ),
        Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentText(text=select_image_prompt2),
                        ContentImage(image=image1),
                        ContentImage(image=image2),
                    ]
                )
            ],
            target=["2"],
            metadata={"tuple_id": tuple_id, "type": "imagescore"},
        ),
    ]


def get_dataset() -> Dataset:
    return hf_dataset(
        "nilshoehing/rocketsciencebench",
        split="train",
        sample_fields=record_to_sample_rocketscience,
        revision="35a8cf32237c9469a47a226d620f03b9c6b1838c",  # specific commit for consistency
    )
