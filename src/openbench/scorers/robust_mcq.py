"""Robust multiple choice answer extraction scorer."""

import re
from typing import Optional

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


# Adapted from https://github.com/openai/gpt-oss
# Comprehensive patterns for extracting MCQ answers, ordered by priority
MCQ_PATTERNS = [
    # 0) Markdown-wrapped "Answer(s)" with letter
    re.compile(
        r"""(?ix)                   # case-insensitive, ignore-space
        (?:\*{1,2}|_{1,2})          # leading *…*  or _…_
        Answer[s]?                  #   Answer or Answers
        \s*[:\-–]?                  #   optional separator
        (?:\*{1,2}|_{1,2})          # closing wrapper
        \s*                         # optional space
        ([ABCD])\b                  # the actual letter
        """,
        re.X,
    ),
    # 0.1) Answer at start of line with various formats
    re.compile(
        r"""(?ix)           # ignore case, allow verbose mode
        ^\s*                        # optional leading whitespace
        (?:\*{1,2}|_{1,2})?         # optional markdown wrapper
        Answer:?                    # the word 'answer' with optional colon
        (?:\*{1,2}|_{1,2})?         # optional markdown wrapper again
        \s*:?\s*                    # optional colon with optional spaces
        (?:\*{1,2}|_{1,2})?         # optional markdown wrapper before letter
        ([ABCD])                    # capture the letter
        (?:\*{1,2}|_{1,2})?         # optional markdown wrapper after letter
        \s*                         # optional trailing whitespace
    """,
        re.MULTILINE,
    ),
    # 1) Answer: (C) or Answers: (B)
    re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)"),
    # 2) Answer: C or Answers – D
    re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b"),
    # 3) Option B or Choice: C
    re.compile(r"(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b"),
    # 4) LaTeX \boxed{...A...}
    re.compile(r"(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}", re.MULTILINE),
    # 5) LaTeX \boxed{\textbf{...C...}}
    re.compile(
        r"(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
    ),
    # 6) LaTeX \boxed{\text{...C...}}
    re.compile(
        r"(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
    ),
    # 7) Bare parentheses or brackets: (A) [B]
    re.compile(r"(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])"),
    # 8) Markdown-wrapped: *A* **B** _C_ __D__
    re.compile(
        r"(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])"
    ),
    # 9) LaTeX \textbf{...C...}
    re.compile(r"(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}"),
    # 10) Markdown-wrapped answer with description: **D) …**
    re.compile(r"""(?x)            # ignore whitespace in pattern
        (?<![A-Za-z0-9])            # not preceded by word-char
        (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
        \s*([ABCD])\)               # capture letter plus ")"
        [^*_\n]+?                   # some text inside wrapper
        (?:\*{1,2}|_{1,2})          # closing wrapper
        (?![A-Za-z0-9])             # not followed by word-char
    """),
    # 11) Final fallback: line that's exactly "A", "B.", "C)", "**D**", etc.
    re.compile(
        r"""(?x)^\s*
        (?:\*{1,2}|_{1,2})?         # optional markdown wrapper
        ([ABCD])                    # capture group for letter
        (?:\*{1,2}|_{1,2})?         # optional closing markdown
        \s*[\.\)\-–:]?              # optional separator after the letter
        \s*.*$                      # allow any following text
    """,
        re.MULTILINE,
    ),
]


def extract_mcq_answer(text: str) -> Optional[str]:
    """
    Extract multiple choice answer (A, B, C, or D) from text using comprehensive patterns.

    Searches through multiple regex patterns to find answer declarations in various
    formats including markdown, LaTeX, and plain text. Patterns are ordered by
    specificity and reliability.


    Args:
        text: The text to search for an answer

    Returns:
        Single letter A, B, C, or D if found, otherwise the first character
        of the text (after removing markdown) as a fallback
    """
    matches = []

    # Try all patterns and collect matches with priority
    for priority, pattern in enumerate(MCQ_PATTERNS):
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            if letter in "ABCD":
                matches.append((priority, match, letter))

    # Sort by priority (lower is better) and match length (longer is better)
    matches.sort(key=lambda triple: (triple[0], -len(triple[1].group(0))))

    # Return the best match if found
    if matches:
        return matches[0][2]

    # Final fallback: return first character after stripping markdown
    cleaned = text.removeprefix("**").strip()
    if cleaned and cleaned[0].upper() in "ABCD":
        return cleaned[0].upper()

    return None


@scorer(metrics=[accuracy(), std(), stderr()])
def robust_mcq_scorer() -> Scorer:
    """
    A robust scorer for multiple choice questions with comprehensive pattern matching.

    This scorer uses multiple regex patterns to extract MCQ answers from various
    formats including markdown, LaTeX, and plain text. It's more robust than
    simple pattern matching and handles edge cases better.

    Returns:
        Scorer: A scoring function that returns a Score with:
            - value: CORRECT if extracted answer matches target, INCORRECT otherwise
            - answer: The extracted answer if found
            - explanation: Details about the extraction method used
    """

    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_mcq_answer(state.output.completion)

        if extracted is None:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="No multiple choice answer found in response",
            )

        is_correct = extracted == target.text.strip().upper()

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extracted,
            explanation=f"Extracted '{extracted}' from response, target was '{target.text}'",
        )

    return score
