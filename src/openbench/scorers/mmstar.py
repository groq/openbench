from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target
from inspect_ai.solver import TaskState
from typing import Callable
import json

from openbench.metrics.mmstar import mmstar_metrics


@scorer(metrics=[accuracy(), stderr(), mmstar_metrics()])
def mmstar_scorer() -> Callable:
    """MMStar scorer"""

    async def score(state: TaskState, target: Target) -> Score:
        target_text = (target.text).strip().upper().replace("\n", "")

        model_responses = json.loads(state.output.completion)
        with_vision_answer = (
            model_responses.get("with_vision_answer", "")
            .strip()
            .upper()
            .replace("\n", "")
        )
        without_vision_answer = (
            model_responses.get("without_vision_answer", "")
            .strip()
            .upper()
            .replace("\n", "")
        )
        text_base_answer = (
            model_responses.get("text_base_answer", "")
            .strip()
            .upper()
            .replace("\n", "")
        )

        # simple MCQ match scoring
        sv = 1.0 if with_vision_answer == target_text else 0.0
        swv = 1.0 if without_vision_answer == target_text else 0.0
        st = 1.0 if text_base_answer == target_text else 0.0

        return Score(
            value=sv,
            answer=with_vision_answer,
            metadata={
                "category": state.metadata["category"],
                "subcategory": state.metadata["subcategory"],
                "image_path": state.metadata["image_path"],
                "question": state.metadata["question"],
                "with_vision_score": sv,
                "without_vision_score": swv,
                "text_base_score": st,
            },
        )

    return score
