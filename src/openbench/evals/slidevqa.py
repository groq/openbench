import base64
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import multiple_choice
from inspect_ai import Task, task
from inspect_ai.scorer import choice
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage


def convert_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to data URI string for multimodal input."""
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    image_uri = f"data:image/png;base64,{b64_str}"
    return image_uri


def record_to_sample(record: dict) -> Sample:
    # TODO: Implement record transformation for SlideVQA dataset
    input_list = [
        ChatMessageUser(content=[ContentText(text=record["question"])]),
    ]

    for i in range(1, 21):  # Process slide images
        image_bytes = record[f"page_{i}"]["bytes"]
        image_uri = convert_image_to_base64(image_bytes)
        input_list.append(ContentImage(image=image_uri)) # type: ignore

    return Sample(
        id=record["id"],
        input=input_list,  # type: ignore
        target=record["answer"],
    )


@task
def slidevqa():
    # TODO: Configure SlideVQA dataset and evaluation
    return Task(
        dataset=hf_dataset(
            "NTT-hil-insight/SlideVQA", split="test", sample_fields=record_to_sample
        ),
        solver=[multiple_choice()],
        scorer=choice(),
    )
