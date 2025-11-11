from inspect_ai import task, Task
from openbench.datasets.rocketscience import get_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import pattern
from openbench.metrics.rocketscience import rocketscience_metrics


@task
def rocketscience() -> Task:
    """RocketScience: Testing spatial understanding in vision-language models.

    RocketScience is a contrastive benchmark that tests understanding of spatial relations
    in Vision Language Models. Each dataset item consisting of image1, image2, text1, text2 generates 4 evaluation samples:
    1. Select best text for image1 (expected: "1")
    2. Select best text for image2 (expected: "2")
    3. Select best image for text1 (expected: "1")
    4. Select best image for text2 (expected: "2")

    We calculacte three metrics:
    - Text Score: per dataset item text_score is 1.0 if both text selection tasks are correct, else 0.0. The text scores for all dataset items are then averaged to get the final text score.
    - Image Score: per dataset item image_score is 1.0 if both image selection tasks are correct, else 0.0. The image scores for all dataset items are then averaged to get the final image score.
    - Group Score: per dataset item group_score is 1.0 if both text and both image selection tasks are correct, else 0.0. The group scores for all dataset items are then averaged to get the final group score.

    Based on the repository: https://github.com/nilshoehing/rocketscience/
    Dataset: https://huggingface.co/datasets/nilshoehing/rocketsciencebench

    If you limit the number of samples in the dataset (with --limit), ensure it's a multiple of 4 to maintain complete evaluation groups.

    Returns:
        Task configured for RocketScience evaluation following original methodology
    """

    return Task(
        dataset=get_dataset(),
        solver=generate(),
        scorer=pattern(pattern=r".*?([12])\D*$"),
        name="rocketscience",
        metrics=[
            rocketscience_metrics(),
        ],
    )
