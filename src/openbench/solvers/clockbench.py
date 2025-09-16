"""Clockbench solver."""

from __future__ import annotations

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageSystem,
    ContentText,
    ContentImage,
)
import json
import re


def try_parse_json(response_text: str) -> dict | None:
    """
    Robust JSON parsing from potentially messy model responses.
    Handles:
    - Perfect JSON: {"valid": true, "hours": 12}
    - Markdown wrapped: ```json\n{...}\n```
    - Extra text: "The answer is: {json}"
    - Malformed JSON (trailing commas, etc.)
    """
    if not response_text:
        return None

    # try direct parsing first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # look for JSON block in markdown
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # look for any JSON-like structure
    json_match = re.search(r"\{[^{}]*\}", response_text)
    if json_match:
        try:
            # clean up common issues
            json_str = json_match.group(0)
            # remove trailing commas before closing braces/brackets
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


@solver
def clockbench_solver():
    """Multi-turn solver that mirrors the original ask_questions() implementation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # extract questions and image
        q_time = state.metadata["question"]["time"].strip()
        q_shift = state.metadata["question"]["shift"].strip()
        q_angle = state.metadata["question"]["angle"].strip()
        q_zone = state.metadata["question"]["zone"].strip()
        img_ref = state.metadata["image_data_uri"]  # Pre-converted image

        # start fresh conversation - clear any existing messages
        state.messages = []

        # system message (from original implementation)
        system_msg = ChatMessageSystem(
            content="Be precise. When JSON is requested, reply with ONLY that JSON (no preface, no code block)."
        )
        state.messages.append(system_msg)

        # --- time question ---
        # note: original implementation only adds image to the first message
        user_msg_1 = ChatMessageUser(
            content=[ContentText(text=q_time), ContentImage(image=img_ref)]
        )
        state.messages.append(user_msg_1)

        # generate response
        result_1 = await generate(state)
        a1_text = result_1.output.completion
        a1_parsed = try_parse_json(a1_text)

        # --- shift question ---
        user_msg_2 = ChatMessageUser(content=q_shift)
        state.messages.append(user_msg_2)

        # generate response
        result_2 = await generate(state)
        a2_text = result_2.output.completion
        a2_parsed = try_parse_json(a2_text)

        # --- angle question ---
        user_msg_3 = ChatMessageUser(content=q_angle)
        state.messages.append(user_msg_3)

        # generate response
        result_3 = await generate(state)
        a3_text = result_3.output.completion
        a3_parsed = try_parse_json(a3_text)

        # --- time zone question ---
        user_msg_4 = ChatMessageUser(content=q_zone)
        state.messages.append(user_msg_4)

        # generate response
        result_4 = await generate(state)
        a4_text = result_4.output.completion
        a4_parsed = try_parse_json(a4_text)

        # store all responses in output as JSON (follows Inspect convention))
        state.output.completion = json.dumps(
            {
                "time": a1_parsed if a1_parsed is not None else a1_text,
                "shift": a2_parsed if a2_parsed is not None else a2_text,
                "angle": a3_parsed if a3_parsed is not None else a3_text,
                "zone": a4_parsed if a4_parsed is not None else a4_text,
            },
            indent=2,
        )

        return state

    return solve
