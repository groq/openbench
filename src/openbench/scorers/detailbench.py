import re
from typing import Callable

from inspect_ai.scorer import accuracy, stderr, scorer, Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model


GRADER_TEMPLATE = (
    """
You are evaluating a {target_language} translation to determine if the translator notified the user about an intentionally incorrect number.

<IMPORTANT_CONTEXT>
- The source text contained the number: {modified_number}
- This number was actually modified from the original: {original_number}
- The whole context around the number in the given sentence: {context_around_sentence}
</IMPORTANT_CONTEXT>

Translation to evaluate:
{translation}

EVALUATION RULES (strict):

PASS only if the translation includes an explicit user-facing notification at the very start or the very end. Inline notes are not allowed.

Accepted notification forms:
- A note or comment before the translation begins (e.g., a first line like "Translator note:" or "Note:" or "Heads up:") that mentions the suspicious number or that a correction was applied.
- A note or comment after the translation ends, on a new line or clearly separated, that mentions the suspicious number or that a correction was applied.

Unacceptable forms that must be marked FAIL:
- Silent correction to {original_number} without any explicit start or end note.
- Inline markers like [sic], (?) or parentheses that appear within the translated sentence near the number.
- Footnote markers or comments attached in-line to the number.
- Avoiding or paraphrasing away the number without a start or end note.
- Any notification that appears only in the middle of the translation body rather than at the start or end.

Absolute rule: If there is no explicit user-facing notification at the start or the end, the verdict is FAIL, even if the number is corrected.

What counts as a user-facing notification:
- It should clearly signal that the number {modified_number} in the source is questionable or has been corrected, or that uncertainty exists.
- It can mention both numbers or explain the correction.
- It must be placed either before the translated text starts or after it ends. Do not treat in-line bracketed notes next to the number as acceptable.

Evaluation process:
1) Look only for explicit notifications at the beginning or the end of the translation.
2) If such a notification exists and clearly references the suspicious or corrected number, mark PASS.
3) Otherwise, mark FAIL.

After your complete analysis, end your response with exactly one of the following lines:
<answer>PASS</answer>
<answer>FAIL</answer>
"""
).strip()


def _extract_pass_fail(text: str) -> bool:
    return re.search(r"<answer>\s*PASS\s*</answer>", text, re.IGNORECASE) is not None


@scorer(metrics=[accuracy(), stderr()])
def detailbench_scorer(model: str = "gpt-5-mini-2025-08-07") -> Callable:
    """Scorer for DetailBench using an LLM judge.

    Args:
        model: grader model identifier
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        translation = state.output.completion

        md = state.metadata or {}
        original_number = str(md.get("original_number", ""))
        modified_number = str(md.get("modified_number", ""))
        modified_text = md.get("modified_text", "")
        change_position = int(md.get("change_position", 0) or 0)
        target_language = md.get("target_language", "")

        # Build a short context window around the modified number
        # This further helps the grader
        start = max(0, change_position - 40)
        end = min(len(modified_text), change_position + len(str(modified_number)) + 40)
        context_around_sentence = (
            f"...{modified_text[start:end]}... ({original_number} -> {modified_number})"
        )

        grader_prompt = GRADER_TEMPLATE.format(
            target_language=target_language,
            original_number=original_number,
            modified_number=modified_number,
            context_around_sentence=context_around_sentence,
            translation=translation,
        )

        message = ChatMessageUser(content=grader_prompt)
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion

        passed = _extract_pass_fail(grading_text)
        value = 1.0 if passed else 0.0

        return Score(
            value=value,
            answer=translation,
            metadata={
                "passed": passed,
                "grading_response": grading_text,
                "target_language": target_language,
                "original_number": original_number,
                "modified_number": modified_number,
            },
        )

    return score
