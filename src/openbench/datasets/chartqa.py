import base64
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage


def generate_user_prompt(query: str) -> str:
    """
    Attribution: EleutherAI/lm-evaluation-harness prompt

    Source:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/chartqa/chartqa.yaml
    """
    PROMPT = f"""
    {query}
      Analyze the image and question carefully, using step-by-step reasoning.
      First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
      Final Answer: <answer>
      where <answer> follows the following instructions:
      - <answer> should should be a single phrase or number.
      - <answer> should not paraphrase or reformat the text in the image.
      - If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
      - If the question is a Yes/No question, <answer> should be Yes/No.
      - If <answer> is a number, it should not contain any units.
      - If <answer> is a percentage, it should include a % sign.
      - If <answer> is an entity, it should include the full label from the graph.
      IMPORTANT: Remember, to end your answer with Final Answer: <answer>.
    """
    return PROMPT


def record_to_sample(record: dict) -> Sample:
    input_message = generate_user_prompt(record["query"])
    b64_str = base64.b64encode(record["image"]["bytes"]).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"
    input_list = [
        ChatMessageUser(content=[ContentText(text=input_message)]),
        ChatMessageUser(content=[ContentImage(image=data_uri)]),
    ]
    return Sample(
        input=input_list,  # type: ignore
        target=record["label"],
    )


def load_dataset(split: str = "test"):
    dataset = hf_dataset(
        "HuggingFaceM4/ChartQA", split=split, sample_fields=record_to_sample
    )
    return dataset
