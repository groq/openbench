import re
from inspect_ai.solver import TaskState
from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    std,
    stderr,
    Target,
    Score,
)
from openbench.metrics.grouped import grouped
from openbench.utils.text import (
    strip_md_latex,
    normalize_mcq_answer,
)


@scorer(metrics=[grouped(group_key="category", metric=[accuracy(), stderr(), std()])])
def mmlu_pro_eval_scorer() -> Callable:
    async def score(state: TaskState, target: Target) -> Score:
        response_text = strip_md_latex(state.output.completion)
        extracted_answer = None

        match = re.search(r"Answer:\s*(.*)$", response_text, re.MULTILINE)
        if match:
            extracted_answer = normalize_mcq_answer(match.group(1))

        if extracted_answer == target.text:
            return Score(value="C", answer=extracted_answer)
        else:
            return Score(value="I", answer=extracted_answer)

    return score
