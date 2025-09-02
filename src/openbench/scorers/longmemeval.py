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


LONGMEMEVAL_GRADER_TEMPLATE = """
Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
In some cases, 0 and "I do not have an answer" are considered to be both correct. 
If both responses say that there is no answer, this should be judged as true.
If the correct answer contains an answer, but the response abstains from answering, this should be judged as false.

Question: {question}

CORRECT answer: {correct_answer}

Response to judge: {output}

Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
""".strip()


@metric
def longmemeval_metrics() -> Metric:
    """Calculate LongMemEval specific metrics: F1 and accuracy_given_attempted."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {
                "is_correct": 0.0,
                "is_incorrect": 0.0,
                "is_not_attempted": 0.0,
                "is_given_attempted": 0.0,
                "accuracy_given_attempted": 0.0,
                "f1": 0.0,
            }

        grade_counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}

        for sample_score in scores:
            metadata = sample_score.score.metadata
            grade = metadata.get("grade", "").lower() if metadata else ""
            if grade in grade_counts:
                grade_counts[grade] += 1

        total = len(scores)
        is_correct = grade_counts["correct"] / total
        is_incorrect = grade_counts["incorrect"] / total
        is_not_attempted = grade_counts["not_attempted"] / total
        is_given_attempted = is_correct + is_incorrect

        accuracy_given_attempted = (
            is_correct / is_given_attempted if is_given_attempted > 0 else 0.0
        )

        f1 = (
            2
            * accuracy_given_attempted
            * is_correct
            / (accuracy_given_attempted + is_correct)
            if (accuracy_given_attempted + is_correct) > 0
            else 0.0
        )

        return {
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted,
            "is_given_attempted": is_given_attempted,
            "accuracy_given_attempted": accuracy_given_attempted,
            "f1": f1,
        }

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), longmemeval_metrics()])
def longmemeval_scorer(model: str) -> Callable:
    """LongMemEval scorer using model grading."""

    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        question = state.metadata.get("question", "")

        predicted_answer = state.output.completion

        grader_prompt = LONGMEMEVAL_GRADER_TEMPLATE.format(
            question=question, correct_answer=target.text, output=predicted_answer
        )

        message = ChatMessageUser(content=grader_prompt)

        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion.strip().lower()

        if "true" in grading_text and "false" not in grading_text:
            grade_name = "correct"
            score_value = 1.0
        elif "false" in grading_text:
            grade_name = "incorrect"
            score_value = 0.0
        else:
            grade_name = "not_attempted"
            score_value = 0.0

        return Score(
            value=score_value,
            answer=predicted_answer,
            metadata={
                "grade": grade_name,
                "grading_response": grading_text,
                "question": question,
            },
        )

    return score
