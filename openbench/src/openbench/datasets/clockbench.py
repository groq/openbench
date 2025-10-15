from __future__ import annotations

from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from openbench.utils.image import detect_image_mime_type
from inspect_ai.model import ContentImage, ChatMessageUser
import base64
import json
from inspect_ai.dataset import hf_dataset


def record_to_sample(record: dict) -> Sample:
    """Convert a Clockbench record to a Sample for multi-turn conversation."""

    # handle the image data - stored as raw bytes in parquet
    image_bytes = record["image"]

    if isinstance(image_bytes, dict) and "bytes" in image_bytes:
        image_bytes = image_bytes["bytes"]
    elif isinstance(image_bytes, bytes):
        image_bytes = image_bytes
    else:
        image_bytes = None

    # convert to base64 data URI with proper MIME type detection
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = detect_image_mime_type(image_bytes)
    data_uri = f"data:{mime_type};base64,{base64_image}"

    # create initial input with just image - solver will add questions dynamically
    image_content = ContentImage(image=data_uri)

    question = {
        "time": record["question_time"],
        "shift": record["question_shift"],
        "angle": record["question_angle"],
        "zone": record["question_zone"],
    }

    # parse target fields from JSON strings
    target = {
        "time": json.loads(record["target_time"]),
        "shift": json.loads(record["target_shift"]),
        "angle": json.loads(record["target_angle"]),
        "zone": json.loads(record["target_zone"]),
    }

    return Sample(
        id=record["id"],
        input=[ChatMessageUser(content=[image_content])],
        target="target",  # placeholder for target, actual ground truthpulled from metadata in solver
        metadata={
            "image_data_uri": data_uri,
            "question": question,
            "target": target,
        },
    )


def get_clockbench_dataset() -> Dataset:
    """Load the Clockbench dataset.

    This dataset is structured to work with a multi-turn solver that mirrors
    the original ask_questions() implementation:

    1. System message: "Be precise. When JSON is requested, reply with ONLY that JSON..."
    2. Turn 1: question_time + image → assistant response → add to messages
    3. Turn 2: question_shift (text only) → assistant response → add to messages
    4. Turn 3: question_angle (text only) → assistant response → add to messages
    5. Turn 4: question_zone (text only) → assistant response

    Each question builds on the conversation context from previous turns.
    """
    # load the dataset from HuggingFace (following HLE pattern)
    dataset = hf_dataset(
        "nmayorga7/clockbench",
        split="train",
        sample_fields=record_to_sample,
    )

    # convert to list for MemoryDataset
    samples = list(dataset)

    return MemoryDataset(samples=samples, name="clockbench")
