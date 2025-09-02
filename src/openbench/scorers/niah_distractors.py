from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model


NIAH_DISTRACTORS_GRADER_TEMPLATE = """
Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
You must only respond with "true" or "false".
If the response is partially incorrect, such as a typo, respond with "false".
If the repsonse contains a snippet of text or additional supporting information, while still maintaining the correct answer without changing the meaning, respond with "true".
If the response starts with anything like "here is the most relevant information in the documents: ", respond with "true". This is fine as long as the following content aligns with the correct answer.

Question: {question}

CORRECT answer: {correct_answer}

Response to judge: {output}

Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
"""


@metric
def niah_distractors_metrics() -> Metric:
    """Calculate NIAH distractors metrics by input length."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {}

        bins = {}
        for sample_score in scores:
            metadata = sample_score.score.metadata
            if not metadata:
                continue

            sample_id = metadata.get("id", "")
            if not sample_id or "_" not in sample_id:
                continue

            token_bin = sample_id.split("_")[0]
            if token_bin not in bins:
                bins[token_bin] = {"correct": 0, "total": 0}

            bins[token_bin]["total"] += 1
            if sample_score.score.as_float() > 0.5:
                bins[token_bin]["correct"] += 1

        accuracy_by_bin = {}
        for bin_name, counts in bins.items():
            accuracy = (
                counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            )
            accuracy_by_bin[f"accuracy_bin_{bin_name}"] = accuracy

        return accuracy_by_bin

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), niah_distractors_metrics()])
def niah_distractors_scorer(model: str) -> Callable:
    """NIAH distractors scorer using model grading with bin-based metrics."""

    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        question = state.metadata.get("question")
        sample_id = state.metadata.get("id")

        predicted_answer = state.output.completion

        grader_prompt = NIAH_DISTRACTORS_GRADER_TEMPLATE.format(
            question=question, correct_answer=target.text, output=predicted_answer
        )

        message = ChatMessageUser(content=grader_prompt)

        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion.strip().lower()

        if "true" in grading_text and "false" not in grading_text:
            score_value = 1.0
        else:
            score_value = 0.0

        return Score(
            value=score_value,
            answer=predicted_answer,
            metadata={
                "id": sample_id,
                "grading_response": grading_text,
                "question": question,
            },
        )

    return score
