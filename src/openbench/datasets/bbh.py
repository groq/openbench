import requests  # type: ignore
from inspect_ai.dataset import Sample, MemoryDataset


def load_dataset(subtask: str, cot: str) -> MemoryDataset:
    if cot.lower() not in ["true", "false"]:
        print(f"cot must be either 'True' or 'False', got {cot}. Defaulting to 'True'.")
        cot = "True"

    if subtask is None:
        print(
            f"subtask is required, got {subtask}. Defaulting to 'sports_understanding'."
        )
        subtask = "sports_understanding"

    if cot.lower() == "true":
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/refs/heads/main/code-davinci-002-outputs/code-davinci-002-cot/{subtask}_few_shot_template_0-255000.json"
    elif cot.lower() == "false":
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/refs/heads/main/code-davinci-002-outputs/code-davinci-002-direct/{subtask}_few_shot_template_0-255000.json"
    else:
        raise ValueError("cot must be either 'True' or 'False'")

    response = requests.get(url)
    data = response.json()

    question_dataset = []
    for example in data["outputs"]:
        question_dataset.append(
            Sample(
                input=[
                    {  # type: ignore
                        "role": "user",
                        "content": example["input"],
                    },
                ],
                target=example["target"],
            )
        )

    dataset = MemoryDataset(question_dataset)

    return dataset
