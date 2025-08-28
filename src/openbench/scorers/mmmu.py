"""MMMU (Massive Multi-discipline Multimodal Understanding) scorer."""

import re
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState
from openbench.utils.text import (
    strip_md_latex,
    normalize_mcq_answer,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
)
from openbench.metrics.grouped import grouped


def extract_mmmu_answer(text: str) -> str:
    """Extract multiple choice answer (A, B, C, D) from model output using multilingual patterns."""
    if not text:
        return ""

    # clean the text of markdown/latex formatting
    response_text = strip_md_latex(text)

    extracted_answer = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_mcq_answer(match.group(1))
            break

    # Fallback to simpler patterns if multilingual extraction fails
    if not extracted_answer:
        fallback_patterns = [
            r"(?:answer|choice|option|select).*?([ABCD])\b",
            r"\b([ABCD])\)",
            r"\(([ABCD])\)",
            r"^([ABCD])(?:\.|:|\s|$)",
            r"\b([ABCD])(?:\.|:|\s|$)",
            r"(?:the )?answer is ([ABCD])",
            r"(?:i choose|i select) ([ABCD])",
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).upper()
                break

    # Final fallback: look for any A-D letters
    if not extracted_answer:
        letters = re.findall(r"\b([ABCD])\b", response_text.upper())
        if letters:
            extracted_answer = letters[0]

    return extracted_answer or ""


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(group_key="subfield", metric=[accuracy(), stderr()]),
        grouped(group_key="topic_difficulty", metric=[accuracy(), stderr()]),
    ]
)
def mmmu_scorer() -> Scorer:
    """MMMU scorer for multiple choice questions."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            extracted_answer = extract_mmmu_answer(state.output.completion)
            target_answer = target.text.strip().upper()

            # Check if extracted answer matches target
            is_correct = extracted_answer == target_answer

            # Get additional metadata for analysis
            metadata = state.metadata if isinstance(state.metadata, dict) else {}
            subfield = metadata.get("subfield", "")
            difficulty = metadata.get("topic_difficulty", "")
            num_images = metadata.get("num_images", 0)

            return Score(
                value=1.0 if is_correct else 0.0,
                answer=extracted_answer,
                metadata={
                    "extracted_answer": extracted_answer,
                    "target_answer": target_answer,
                    "subfield": subfield,
                    "difficulty": difficulty,
                    "num_images": num_images,
                    "is_correct": is_correct,
                    "raw_output": state.output.completion,
                },
            )
        except Exception as e:
            # Log the error and return a zero score to prevent evaluation failure
            print(
                f"Warning: Error scoring MMMU sample {getattr(state, 'sample_id', 'unknown')}: {e}. "
                f"Returning zero score."
            )
            return Score(
                value=0.0,
                answer="",
                metadata={
                    "extracted_answer": "",
                    "target_answer": target.text.strip().upper()
                    if target and target.text
                    else "",
                    "error": str(e),
                    "raw_output": state.output.completion if state.output else "",
                },
            )

    return score
