"""IFBench scorer."""

from typing import Any, Dict, List

from inspect_ai.scorer import Score, Target, scorer, Scorer
from inspect_ai.solver import TaskState

from openbench.ifbench.instructions_registry import INSTRUCTION_DICT
from openbench.metrics.ifeval import ifeval_metrics


def _clean_kwargs_list(raw_kwargs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for kwargs in raw_kwargs:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        cleaned.append(filtered)
    return cleaned


@scorer(metrics=[ifeval_metrics()])
def ifbench_scorer() -> Scorer:
    """Score IFBench strict/loose instruction following."""

    async def score(state: TaskState, target: Target) -> Score:
        instruction_list = state.metadata.get("instruction_id_list", [])
        raw_kwargs_list = state.metadata.get("kwargs", [])

        if not instruction_list:
            return Score(value=0.0, explanation="No instructions found")

        kwargs_list = _clean_kwargs_list(raw_kwargs_list)
        response = state.output.completion
        input_prompt = getattr(state, "input_text", "") or str(state.input)

        strict_following_list = []
        loose_following_list = []

        for idx, instruction_id in enumerate(instruction_list):
            instruction_cls = INSTRUCTION_DICT.get(instruction_id)
            if instruction_cls is None:
                return Score(
                    value=0.0,
                    explanation=f"Unknown instruction id: {instruction_id}",
                )
            instruction = instruction_cls(instruction_id)

            kwargs = kwargs_list[idx] if idx < len(kwargs_list) else {}
            instruction.build_description(**kwargs)

            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=input_prompt)

            strict_followed = bool(response.strip()) and instruction.check_following(
                response
            )
            strict_following_list.append(strict_followed)

            r = response.split("\n")
            response_remove_first = "\n".join(r[1:]).strip()
            response_remove_last = "\n".join(r[:-1]).strip()
            response_remove_both = "\n".join(r[1:-1]).strip()
            revised_response = response.replace("*", "")
            revised_response_remove_first = response_remove_first.replace("*", "")
            revised_response_remove_last = response_remove_last.replace("*", "")
            revised_response_remove_both = response_remove_both.replace("*", "")

            all_responses = [
                response,
                revised_response,
                response_remove_first,
                response_remove_last,
                response_remove_both,
                revised_response_remove_first,
                revised_response_remove_last,
                revised_response_remove_both,
            ]

            loose_followed = any(
                candidate.strip() and instruction.check_following(candidate)
                for candidate in all_responses
            )
            loose_following_list.append(loose_followed)

        explanations = [
            f"[S:{'✓' if strict_followed else '✗'} L:{'✓' if loose_followed else '✗'}] {instruction_id}"
            for instruction_id, strict_followed, loose_followed in zip(
                instruction_list,
                strict_following_list,
                loose_following_list,
                strict=True,
            )
        ]

        return Score(
            value=1.0 if all(strict_following_list) else 0.0,
            answer=response,
            explanation="\n".join(explanations),
            metadata={
                "strict_follow_instruction_list": strict_following_list,
                "loose_follow_instruction_list": loose_following_list,
                "instruction_id_list": instruction_list,
            },
        )

    return score
