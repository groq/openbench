from inspect_ai import task, Task
from openbench.utils.mcq import MCQEval, MCQSample


def record_to_mcq_sample(record: dict[str, str]) -> MCQSample:
    """Convert a MMLU Pro record to an OpenBench MCQSample."""
    prompt_list = [
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is a single uppercase letter.",
        "",
        f"{record['question']}",
        "",
    ]

    for i, option in enumerate(record["options"]):
        letter = chr(ord("A") + i)
        prompt_list.append(f"{letter}) {option}")

    prompt_str = "\n".join(prompt_list)

    return MCQSample(
        input=prompt_str,
        target=record["answer"],
        metadata={
            "category": record["category"],
            "src": record["src"],
        },
    )


@task
def mmlu_pro(split="test") -> Task:
    """Evaluate the MMLU Pro dataset. MCQ Abstracted."""
    return MCQEval(
        name="mmlu_pro",
        dataset_path="TIGER-Lab/MMLU-Pro",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        group_keys=["category"],
    )
