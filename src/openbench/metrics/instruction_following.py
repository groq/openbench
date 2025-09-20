"""Instruction Following metrics implementation."""

from inspect_ai.scorer import Metric, Value, SampleScore, metric


@metric
def instruction_following_metrics() -> Metric:
    """Calculate detailed instruction following metrics matching original IFEval."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        instruction_total = 0
        instruction_correct = 0

        for sample_score in scores:
            metadata = sample_score.score.metadata
            if not metadata:
                continue

            follow_list = metadata["follow_instruction_list"]
            instruction_total += len(follow_list)
            instruction_correct += sum(follow_list)

        return (
            {"instruction_level_accuracy": instruction_correct / instruction_total}
            if instruction_total > 0
            else {}
        )

    return metric_calculator
