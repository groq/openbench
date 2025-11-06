"""
ChartQAPro scorer using official evaluation logic.

Reference: https://github.com/vis-nlp/ChartQAPro/blob/main/evaluate_predictions.py

Implements relaxed correctness metric with:
- MCQ & Fact Checking: Exact match
- Numeric answers: 5% relative error tolerance
- Year answers: Exact match (when flagged)
- Textual answers: ANLS (Average Normalized Levenshtein Similarity)
- List-based answers: Element-wise scoring
"""

from __future__ import annotations

import ast
import re
from typing import Any

from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from openbench.metrics.chartqapro import chartqapro_metrics

# ============================================================================
# Helper Functions (from official evaluation code)
# ============================================================================


def fix_list_format(item: str) -> Any:
    """
    Standardize string representations of lists, adding quotes around elements if missing.

    From official evaluation code.
    """
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> list[str] | None:
    """
    Parse text to a list of strings if possible; strips quotes and whitespace.

    From official evaluation code.
    """
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: str) -> float | None:
    """
    Convert text to float, stripping percent signs. Returns None on failure.

    From official evaluation code.
    """
    try:
        return float(text.strip().strip("%"))
    except ValueError:
        return None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def anls_score(
    prediction: str, gold_labels: list[str], threshold: float = 0.5
) -> float:
    """
    Calculate ANLS score (Average Normalized Levenshtein Similarity).

    Compatible with the anls package used in official evaluation.

    Args:
        prediction: Model prediction
        gold_labels: List of acceptable ground truth answers
        threshold: ANLS threshold (default 0.5)

    Returns:
        ANLS score between 0.0 and 1.0
    """
    pred = prediction.lower().strip()

    # Try all gold labels, take max score
    max_score = 0.0
    for gold in gold_labels:
        gold = gold.lower().strip()

        if pred == gold:
            return 1.0

        distance = levenshtein_distance(pred, gold)
        max_len = max(len(pred), len(gold))

        if max_len == 0:
            score = 1.0
        else:
            normalized_distance = distance / max_len
            if normalized_distance < threshold:
                score = 1.0 - normalized_distance
            else:
                score = 0.0

        max_score = max(max_score, score)

    return max_score


def evaluate_single_answer(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> float:
    """
    Evaluate a single target-prediction pair.

    From official evaluation code.

    Args:
        target: Ground truth answer
        prediction: Model prediction
        max_relative_change: Tolerance for numeric answers (default 5%)

    Returns:
        Score: 1.0 for correct, 0.0 for incorrect, or ANLS score for text
    """
    t = target.strip().strip("%").strip()
    p = prediction.strip().strip("%").strip()

    # Attempt numeric comparison
    t_f = to_float(t)
    p_f = to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0

    # Fallback to ANLS for text
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def extract_final_answer(text: str, prompt_strategy: str = "direct") -> str:
    """
    Extract the final answer from model output based on prompting strategy.

    Args:
        text: Model output text
        prompt_strategy: One of ["direct", "cot", "pot"]

    Returns:
        Extracted answer text
    """
    text = text.strip()

    if prompt_strategy == "cot":
        # CoT format: "The answer is X" - extract X
        # Try multiple patterns in order of specificity
        patterns = [
            # Pattern 1: "The answer is X" - most explicit
            (r"[Tt]he answer is[:\s]+(.+?)(?:\.|$)", True),
            # Pattern 2: "Answer: X" or "Answer is X"
            (r"[Aa]nswer(?:\s+is)?[:\s]+(.+?)(?:\.|$)", True),
            # Pattern 3: Last sentence after a period (fallback)
            (r"\.\s+([^.]+?)(?:\.|$)", False),
        ]

        for pattern, use_match in patterns:
            match = re.search(pattern, text)
            if match and use_match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = answer.rstrip(".")
                # Remove quotes if present
                answer = answer.strip("\"'")

                # Special case for MCQ in CoT: extract just the letter if format is "a) text"
                mcq_match = re.match(r"^([a-dA-D])\s*[\):]", answer)
                if mcq_match:
                    return mcq_match.group(1).lower()

                return answer

        # Fallback: return last sentence/line if no pattern found
        last_line = text.split("\n")[-1].strip().rstrip(".")

        # If last line looks like a full sentence with "is/are", extract the answer
        if " is " in last_line or " are " in last_line:
            # Find the last occurrence of "is/are" and extract what comes after
            for sep in [" is ", " are "]:
                if sep in last_line:
                    idx = last_line.rfind(sep)
                    if idx != -1:
                        answer = last_line[idx + len(sep) :].strip()
                        # First, extract quoted text if present (highest priority)
                        quote_match = re.search(r'"([^"]+)"', answer)
                        if quote_match:
                            return quote_match.group(1).strip().rstrip(".")
                        # Remove common trailing/leading modifiers
                        for pattern in [
                            r"\s+with\s+\d+%?$",  # "with 51%"
                            r"^estimated to be\s+around\s+",  # "estimated to be around"
                            r"^around\s+",  # "around"
                        ]:
                            answer = re.sub(pattern, "", answer).strip()
                        return answer.rstrip(".")

        return last_line

    elif prompt_strategy == "pot":
        # PoT format: Python code - need to execute or extract the result

        # First check if it's just "unanswerable" without code
        if "unanswerable" in text.lower() and "```" not in text:
            return "unanswerable"

        # Extract code from markdown code block if present
        code_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = text.strip()

        # Pre-process the code to remove import statements since we provide modules
        # in the namespace already
        code_lines = code.split("\n")
        filtered_lines = []
        for line in code_lines:
            # Skip import statements - we provide modules in namespace
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                continue
            filtered_lines.append(line)
        code = "\n".join(filtered_lines)

        # Try to execute the code safely (with restrictions)
        try:
            # Import common safe modules that might be used
            import io
            import statistics
            from contextlib import redirect_stdout

            # Create a restricted namespace for execution
            namespace = {
                "__builtins__": {
                    "sum": sum,
                    "len": len,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "print": print,
                    "range": range,
                    "enumerate": enumerate,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                },
                "statistics": statistics,
            }

            # Capture print output
            f = io.StringIO()
            with redirect_stdout(f):
                # Execute the code
                exec(code, namespace, namespace)

            printed_output = f.getvalue().strip()

            # If there was print output, use that
            if printed_output:
                return printed_output

            # Otherwise, try to find the result from the last expression
            # Look for the last non-import, non-comment line
            lines = [
                line.strip()
                for line in code.split("\n")
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("import")
            ]
            if lines:
                last_line = lines[-1]
                # If it's just a variable name, get its value
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", last_line):
                    if last_line in namespace:
                        result = namespace[last_line]
                        return str(result)

            # Fallback: return empty if we can't extract
            return ""

        except Exception:
            # If execution fails, return empty (will score as incorrect)
            return ""

    else:  # direct
        # Direct: answer should already be concise
        # Special case for MCQ: extract just the letter if format is "a) text"
        mcq_match = re.match(r"^([a-dA-D])\s*[\):]", text)
        if mcq_match:
            return mcq_match.group(1).lower()

        return text


def relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: list[str] | None = None,
    always_use_exact_match: bool = False,
) -> float:
    """
    Calculate relaxed correctness between target and prediction.

    From official evaluation code with full support for:
    - List-based answers (element-wise scoring)
    - Year flags (exact match override)
    - MCQ/Fact Checking (exact match)
    - Numeric with tolerance
    - Text with ANLS

    Args:
        target: Ground truth answer (can be list format)
        prediction: Model prediction (can be list format)
        max_relative_change: Tolerance for numeric answers (default 0.05)
        year_flags: List of "YES"/"NO" flags indicating if exact match required
        always_use_exact_match: Force exact matching (for MCQ/Fact Checking)

    Returns:
        Score between 0.0 and 1.0 (averaged across list elements if applicable)
    """
    fixed_t = fix_list_format(target)
    t_list = parse_to_list(str(fixed_t)) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)

    # Expand year_flags if needed
    if year_flags is not None and len(year_flags) < n:
        year_flags = year_flags * n

    # Default year flags if not provided
    if year_flags is None:
        year_flags = ["NO"] * max(len(t_list), len(p_list))

    # Evaluate elements
    scores: list[float] = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            # Model predicted more or fewer elements than expected
            scores.append(0.0)
            continue

        t_item = t_list[idx]
        p_item = p_list[idx]
        flag = year_flags[idx] if idx < len(year_flags) else "NO"
        flag_cond = flag.upper() == "YES"

        if flag_cond or always_use_exact_match:
            # Exact match for years, MCQ, or Fact Checking
            try:
                scores.append(
                    1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0
                )
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# Scorer
# ============================================================================


@scorer(metrics=[accuracy(), stderr(), chartqapro_metrics()])
def chartqapro_scorer():
    """
    ChartQAPro scorer using official evaluation logic.

    Handles:
    - Conversational (only last answer scored)
    - Multi Choice / Fact Checking (exact match)
    - Numeric (5% tolerance, except years which require exact match)
    - Text (ANLS with 0.5 threshold)
    - List-based answers (element-wise scoring)

    Returns:
        Scorer function for ChartQAPro samples
    """

    async def score(state: TaskState, target: Any) -> Score:
        try:
            # Get prediction from solver (already extracted by solver)
            prediction = state.output.completion

            # Extract metadata
            question_type = state.metadata.get("question_type", "")
            year_flags = state.metadata.get("year_flags", [])

            # Normalize prediction (strip periods, newlines like official eval)
            # Note: Extraction is now done in the solver, so completion is already clean
            pred_text = str(prediction).strip(".").strip("\n").strip()

            # Normalize target - extract text from Target object if needed
            if hasattr(target, "text"):
                target_text = str(target.text).strip(".").strip("\n").strip()
            else:
                target_text = str(target).strip(".").strip("\n").strip()

            # For conversational, only use last year flag (per official eval line 79)
            if question_type == "Conversational" and year_flags:
                year_flags_to_use = [year_flags[-1]]
            else:
                year_flags_to_use = year_flags

            # Determine if exact match is required
            always_use_exact_match = question_type in ["Fact Checking", "Multi Choice"]

            # Calculate score using official logic
            score_value = relaxed_correctness_chartqapro(
                target=target_text,
                prediction=pred_text,
                max_relative_change=0.05,
                year_flags=year_flags_to_use,
                always_use_exact_match=always_use_exact_match,
            )

            # Determine scoring method for explanation
            if always_use_exact_match:
                scoring_method = "Exact match"
            elif year_flags_to_use and any(
                f.upper() == "YES" for f in year_flags_to_use
            ):
                scoring_method = "Exact match (year)"
            else:
                # Check if numeric comparison was used
                pred_float = to_float(pred_text)
                target_float = to_float(target_text)
                if pred_float is not None and target_float is not None:
                    scoring_method = "Numeric (5% tolerance)"
                else:
                    scoring_method = "ANLS (Levenshtein similarity)"

            return Score(
                value=score_value,
                answer=pred_text,
                explanation=scoring_method,
                metadata={
                    "sample_id": state.sample_id,
                    "question_type": question_type,
                    "num_questions": state.metadata.get("num_questions", 1),
                    "prediction": pred_text,
                    "target": target_text,
                    "year_flags": year_flags_to_use,
                    "exact_match": always_use_exact_match,
                },
            )

        except Exception as e:
            return Score(
                value=0.0,
                answer=state.output.completion if hasattr(state, "output") else "",
                explanation=f"Error: {str(e)}",
                metadata={
                    "error": str(e),
                    "sample_id": getattr(state, "sample_id", "unknown"),
                    "question_type": state.metadata.get("question_type", "unknown"),
                },
            )

    return score
