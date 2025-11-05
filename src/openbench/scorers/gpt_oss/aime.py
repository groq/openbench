import re

from inspect_ai.scorer import (
    Score,
    Scorer,
    scorer,
    CORRECT,
    INCORRECT,
    Target,
    accuracy,
    std,
    stderr,
)
from inspect_ai.solver import TaskState


def extract_boxed_text(text: str) -> str:
    pattern = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(",")[-1].strip()
    pattern = r"\d+"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""


def normalize_number(s):
    match = re.match(r"\d+", s)
    if not match:
        return None
    return match.group(0)


@scorer(metrics=[accuracy(), stderr(), std()])
def gpt_oss_aime_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response_text = state.output.completion if state.output else ""

        extracted_answer = extract_boxed_text(response_text)
        correct_answer_str = target.text.strip()

        # Normalize the extracted answer
        extracted_normalized = normalize_number(extracted_answer)

        # Try to convert to integer (all AIME answers are integers)
        try:
            if extracted_normalized is None:
                return Score(
                    value=INCORRECT,
                    answer=None,
                    explanation="No valid answer found in response",
                )

            extracted_int = int(extracted_normalized)
        except (ValueError, TypeError):
            return Score(
                value=INCORRECT,
                answer=extracted_answer,
                explanation=f"Could not parse '{extracted_answer}' as integer",
            )

        # Compare with target answer
        try:
            correct_answer = int(correct_answer_str)
        except (ValueError, TypeError):
            return Score(
                value=INCORRECT,
                answer=str(extracted_int),
                explanation=f"Could not parse target '{correct_answer_str}' as integer",
            )

        is_correct = extracted_int == correct_answer

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=str(extracted_int),
            explanation=f"Extracted {extracted_int} from response, target was {correct_answer}",
        )

    return score
