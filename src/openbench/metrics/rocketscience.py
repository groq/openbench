from inspect_ai.scorer import metric, Metric, SampleScore, Value
from collections import defaultdict


def _get_contrastive_scores(scores: list[SampleScore]) -> tuple[float, float, float]:
    """Calculate contrastive score per group, returning dict mapping tuple_id -> score"""

    # gather scores by tuple_id
    grouped_scores = defaultdict(list)
    for score in scores:
        if score.sample_metadata is not None:
            grouped_scores[score.sample_metadata["tuple_id"]].append(score)

    # calculate metrics for each tuple_id
    image_scores = []
    text_scores = []
    group_scores = []
    for _, group in grouped_scores.items():
        image_score_items = []
        text_score_items = []

        # divides the four scores into image and text score items
        for item in group:
            if item.sample_metadata is not None:
                if item.sample_metadata["type"] == "imagescore":
                    image_score_items.append(item.score.value)
                elif item.sample_metadata["type"] == "textscore":
                    text_score_items.append(item.score.value)

        if len(image_score_items) == 2 and len(text_score_items) == 2:
            i0 = image_score_items[0]
            i1 = image_score_items[1]
            t0 = text_score_items[0]
            t1 = text_score_items[1]

            image_scores.append(1.0 if i0 == 1.0 and i1 == 1.0 else 0.0)
            text_scores.append(1.0 if t0 == 1.0 and t1 == 1.0 else 0.0)
            group_scores.append(
                1.0 if t0 == 1.0 and t1 == 1.0 and i0 == 1.0 and i1 == 1.0 else 0.0
            )

    image_score = sum(image_scores) / len(image_scores) if image_scores else 0.0
    text_score = sum(text_scores) / len(text_scores) if text_scores else 0.0
    group_score = sum(group_scores) / len(group_scores) if group_scores else 0.0
    return image_score, text_score, group_score


@metric
def rocketscience_metrics() -> Metric:
    """Calculate all RocketScience scores (image, text, and group scores)"""

    def metric_fn(scores: list[SampleScore]) -> Value:
        image_score, text_score, group_score = _get_contrastive_scores(scores)
        return {
            "image_score": image_score,
            "text_score": text_score,
            "group_score": group_score,
        }

    return metric_fn
