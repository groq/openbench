from inspect_ai.scorer import metric, Metric, SampleScore, Value
from collections import defaultdict
from typing import List


def _get_contrastive_score_per_group(
    scores: list[SampleScore], type: str
) -> List[float]:
    """Calculate contrastive score per group based on type (textscore/imagescore)"""

    # find unique tuple_ids
    grouped_scores = defaultdict(list)
    for score in scores:
        if score.sample_metadata is not None:
            grouped_scores[score.sample_metadata["tuple_id"]].append(score)

    # calculate metrics for each group
    metrics = []
    for group in grouped_scores.values():
        return_scores = [
            s
            for s in group
            if s.sample_metadata is not None and s.sample_metadata["type"] == type
        ]
        if len(return_scores) >= 2:
            score1 = return_scores[0].score.value
            score2 = return_scores[1].score.value
            metrics.append(1.0 if score1 == 1.0 and score2 == 1.0 else 0.0)
    return metrics


@metric
def rocketscience_text_score() -> Metric:
    """Calculate RocketScience Text Score"""

    def metric_fn(scores: list[SampleScore]) -> Value:
        score_per_group = _get_contrastive_score_per_group(scores, type="textscore")
        return sum(score_per_group) / len(score_per_group) if score_per_group else 0.0

    return metric_fn


@metric
def rocketscience_image_score() -> Metric:
    """Calculate RocketScience Image Score"""

    def metric_fn(scores: list[SampleScore]) -> Value:
        score_per_group = _get_contrastive_score_per_group(scores, type="imagescore")
        return sum(score_per_group) / len(score_per_group) if score_per_group else 0.0

    return metric_fn


@metric
def rocketscience_group_score() -> Metric:
    """Calculate RocketScience Group Score (combined text and image scores)"""

    def metric_fn(scores: list[SampleScore]) -> Value:
        textscore = _get_contrastive_score_per_group(scores, type="textscore")
        imagescore = _get_contrastive_score_per_group(scores, type="imagescore")
        score_per_group = [
            1.0 if t == 1.0 and i == 1.0 else 0.0 for t, i in zip(textscore, imagescore)
        ]
        return sum(score_per_group) / len(score_per_group) if score_per_group else 0.0

    return metric_fn
