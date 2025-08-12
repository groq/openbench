"""HealthBench scorer implementation."""

from collections import defaultdict
from typing import Any, Callable, Dict

import numpy as np
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, accuracy, metric, scorer, stderr
from inspect_ai.solver import TaskState

from openbench.utils.text import format_chat_messages, parse_json_from_response


GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def format_rubric(rubric: Dict[str, Any]) -> str:
    """Format a rubric item for display."""
    return f"[{rubric['points']}] {rubric['criterion']}"


@metric
def healthbench_metrics():
    """Calculate HealthBench specific metrics including tag scores."""

    def metric_fn(scores):
        if not scores:
            return {}

        # Aggregate tag scores
        tag_values = defaultdict(list)
        for score in scores:
            if score.metadata:
                for key, value in score.metadata.items():
                    if key.startswith("tag_") and isinstance(value, (int, float)):
                        tag_values[key].append(value)

        # Calculate mean for each tag
        result = {}
        for key, values in tag_values.items():
            result[key] = float(np.clip(np.mean(values), 0, 1))

        return result

    return metric_fn


@scorer(metrics=[accuracy(), stderr(), healthbench_metrics()])
def healthbench_scorer(
    grader_model: str = "openai/gpt-4.1-2025-04-14",
) -> Callable:
    """HealthBench scorer using model grading of rubrics.

    Args:
        grader_model: Model to use for grading rubrics
    """
    model = get_model(grader_model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get rubrics from metadata
        rubrics = state.metadata.get("rubrics", [])
        if not rubrics:
            return Score(value=0.0, explanation="No rubrics found")

        # Get example tags
        example_tags = state.metadata.get("example_tags", [])

        # Build conversation with model's response
        prompt_messages = state.input if isinstance(state.input, list) else []
        convo_with_response = prompt_messages + [
            {"role": "assistant", "content": state.output.completion}
        ]
        convo_str = format_chat_messages(convo_with_response)

        # Grade each rubric
        grading_results = []
        for rubric in rubrics:
            # Format grading prompt
            grader_prompt = GRADER_TEMPLATE.format(
                conversation=convo_str, rubric_item=format_rubric(rubric)
            )

            # Get grading from model
            result = await model.generate([ChatMessageUser(content=grader_prompt)])
            grading_dict = parse_json_from_response(result.completion)

            # Check if we got valid response
            if "criteria_met" in grading_dict and isinstance(
                grading_dict["criteria_met"], bool
            ):
                grading_results.append(grading_dict)
            else:
                # Invalid response format, use default
                grading_results.append(
                    {
                        "criteria_met": False,
                        "explanation": f"Invalid grading response format: {result.completion[:100]}",
                    }
                )

        # Calculate overall score
        total_possible = sum(r["points"] for r in rubrics if r["points"] > 0)
        if total_possible == 0:
            overall_score = 0.0
        else:
            achieved = sum(
                r["points"]
                for r, g in zip(rubrics, grading_results)
                if g.get("criteria_met", False)
            )
            overall_score = float(np.clip(achieved / total_possible, 0, 1))

        # Calculate tag scores
        tag_scores = {}

        # Example-level tags get the overall score
        for tag in example_tags:
            tag_scores[f"tag_{tag}"] = overall_score

        # Rubric-level tags
        rubric_tag_groups = defaultdict(list)
        for rubric, grading in zip(rubrics, grading_results):
            for tag in rubric.get("tags", []):
                rubric_tag_groups[tag].append((rubric, grading))

        for tag, items in rubric_tag_groups.items():
            tag_total = sum(r["points"] for r, _ in items if r["points"] > 0)
            if tag_total > 0:
                tag_achieved = sum(
                    r["points"] for r, g in items if g.get("criteria_met", False)
                )
                tag_scores[f"tag_{tag}"] = float(
                    np.clip(tag_achieved / tag_total, 0, 1)
                )

        # Build readable explanation
        explanations = []
        for rubric, grading in zip(rubrics, grading_results):
            met = grading.get("criteria_met", False)
            exp = grading.get("explanation", "No explanation")
            status = "✓" if met else "✗"
            explanations.append(f"[{status}] {format_rubric(rubric)}\n  {exp}")

        # Sort to show failures first
        explanations.sort(key=lambda x: x.startswith("[✗]"), reverse=True)

        return Score(
            value=overall_score,
            answer=state.output.completion,
            explanation="\n\n".join(explanations),
            metadata={
                "overall_score": overall_score,
                **tag_scores,
            },
        )

    return score
