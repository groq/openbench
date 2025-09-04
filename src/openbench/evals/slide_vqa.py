from inspect_ai import task, Task
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig
from openbench.datasets.slide_vqa import get_dataset
from openbench.scorers.slide_vqa import slide_vqa_scorer


SLIDE_VQA_SYSTEM_PROMPT = """You are an expert at analyzing presentation slides and answering questions about their content. 

When presented with slides and a question:
1. Carefully examine all provided slide images
2. Look for relevant information that answers the question
3. Pay attention to text, charts, graphs, diagrams, and other visual elements
4. Provide a clear, concise answer based on the information shown in the slides
5. If the answer involves numbers or calculations, show your reasoning

Answer directly and precisely based only on what you can see in the slides."""


@task
def slide_vqa(split: str = "test", max_tokens: int = 4096) -> Task:
    """SlideVQA: Visual Question Answering on presentation slides.

    SlideVQA is a dataset for visual question answering on presentation slides,
    requiring models to understand and reason about multi-page slide content
    including text, charts, and visual elements.

    Args:
        split: Dataset split to use ('train', 'test', or 'eval')
        max_tokens: Maximum tokens for model response

    Returns:
        Task configured for SlideVQA evaluation
    """
    return Task(
        dataset=get_dataset(split=split),
        solver=[
            system_message(SLIDE_VQA_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=slide_vqa_scorer(),
        name=f"slide_vqa_{split}",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=max_tokens,
        ),
    )
