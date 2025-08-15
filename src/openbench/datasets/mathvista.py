from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage
import base64


def record_to_sample(record: dict) -> Sample:
    b64_str = base64.b64encode(record["decoded_image"]["bytes"]).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    sample_input = [
        ChatMessageUser(content=[ContentText(text=record["query"])]),
        ChatMessageUser(content=[ContentImage(image=data_uri)]),
    ]

    return Sample(
        input=sample_input,  # type: ignore
        metadata={
            "question_type": record["question_type"],
            "precision": record["precision"],
        },
        choices=record["choices"],
        target=record["answer"],
    )
