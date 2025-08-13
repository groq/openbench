from inspect_ai.dataset import Dataset, csv_dataset, Sample, MemoryDataset
from typing import Any


# Both remove_boxed() and last_boxed_only_string() functions borrowed from:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L53C1-L94C18
def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        target=remove_boxed(
            last_boxed_only_string(record["Answer"]) or record["Answer"]
        ),
    )


def get_dataset(split: str = "math_test") -> Dataset:
    """Load the MATH dataset.

    Args:
        split: Which dataset split to use - "math_test" for full 5000 problems,
               or "math_500_test" for 500 problem subset
    """
    # Load the dataset from OpenAI's blob storage
    dataset = csv_dataset(
        csv_file=f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv",
        sample_fields=record_to_sample,
        auto_id=True,
        name=split,
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name=split)
