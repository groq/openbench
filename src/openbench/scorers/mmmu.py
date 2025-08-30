"""MMMU mixed-type scorer that dispatches based on question_type.

Uses MCQ scoring for multiple-choice items and parser for open-answer items.
Supports grouped metrics across both types.
"""

from typing import Callable, Optional, Sequence, Union, List, Any
import re

from inspect_ai.scorer import scorer, accuracy, stderr, std, Score, Target
from inspect_ai.solver import TaskState

from openbench.scorers.mcq import create_mcq_scorer
from openbench.metrics.grouped import grouped


def _build_metrics(group_keys: Optional[List[str]] = None) -> List[Any]:
    metrics: List[Any] = [accuracy(), stderr(), std()]
    if group_keys:
        for key in group_keys:
            metrics.append(grouped(group_key=key, metric=[accuracy(), stderr(), std()]))
    return metrics


def mmmu_mixed_scorer(
    group_keys: Optional[List[str]] = None,
) -> Callable:
    """Create a scorer that handles both MCQ and open-answer MMMU items.

    Args:
        grader_model: Model used to grade open-answer items.
        group_keys: Optional metadata keys to group metrics by.
    """

    # Instantiate underlying scorers
    mcq_scorer = create_mcq_scorer(group_keys=None)()

    @scorer(metrics=_build_metrics(group_keys))
    def mixed() -> Callable:
        async def score(state: TaskState, target: Target) -> Score:
            qtype = str(state.metadata.get("question_type", "multiple-choice")).lower()
            if qtype == "open":
                # Parse open response and compare with normalized answers
                parsed_predictions = _parse_open_response(state.output.completion or "")
                answer = state.metadata.get("answer", target.text)
                is_correct = _eval_open(answer, parsed_predictions)
                return Score(
                    value=1.0 if is_correct else 0.0,
                    answer=state.output.completion,
                    metadata={"open_eval": True, "pred_list": parsed_predictions},
                )
            return await mcq_scorer(state, target)

        return score

    return mixed


# ------------------------
# Open-answer parsing logic
# ------------------------
def _extract_numbers(text: str) -> list[Union[float, str]]:
    numbers: list[Union[float, str]] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            num = float(match)
            numbers.append(num)
        except ValueError:
            continue
    return numbers


def _normalize_str(value: Union[str, float, int]) -> list[Union[str, float]]:
    if isinstance(value, (float, int)):
        return [float(value)]
    s = str(value).strip().strip(".").strip().lower()
    # Remove extra punctuation/spaces
    s = re.sub(r"\s+", " ", s)
    # Also add a numeric interpretation if it parses
    out: list[Union[str, float]] = []
    try:
        num = float(s.replace(",", ""))
        out.append(float(num))
    except ValueError:
        pass
    if s:
        out.append(s)
    return out


def _parse_open_response(response: str) -> list[Union[str, float]]:
    def get_key_subresponses(resp: str) -> list[str]:
        key_responses: list[str] = []
        resp = resp.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        for index, sub in enumerate(sub_responses):
            local_indicators = list(indicators_of_keys)
            if index == len(sub_responses) - 1:
                local_indicators.extend(["="])
            shortest: Optional[str] = None
            for indicator in local_indicators:
                if indicator in sub:
                    candidate = sub.split(indicator)[-1].strip()
                    if not shortest or len(candidate) < len(shortest):
                        shortest = candidate
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest)
        if not key_responses:
            return [resp]
        return key_responses

    key_responses = get_key_subresponses(response or "")
    pred_list: list[Union[str, float]] = list(key_responses)
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    tmp: list[Union[str, float]] = []
    for item in pred_list:
        tmp.extend(_normalize_str(item))
    pred_list = list(set(tmp))
    return pred_list


def _eval_open(
    gold: Union[str, Sequence[Any]], pred_list: list[Union[str, float]]
) -> bool:
    # Normalize gold answers into comparable forms (strings and floats)
    norm_answers: list[Union[str, float]] = []
    if isinstance(gold, (str, float, int)):
        norm_answers = _normalize_str(gold)
    elif isinstance(gold, Sequence):
        for ans in gold:
            value: Union[str, float, int]
            if isinstance(ans, (str, float, int)):
                value = ans
            else:
                value = str(ans)
            norm_answers.extend(_normalize_str(value))
    else:
        norm_answers = _normalize_str(str(gold))

    correct = False
    for pred in pred_list:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    correct = True
                    break
        else:
            if pred in norm_answers:
                correct = True
        if correct:
            break
    return correct
