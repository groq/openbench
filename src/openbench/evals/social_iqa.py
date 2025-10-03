"""
Social IQA: Social Intelligence Question Answering

This benchmark evaluates models on social and emotional intelligence through multiple-choice
questions about social situations. Models must reason about people's actions, emotions,
mental states, and social norms.

Dataset: social_i_qa
Split: validation

Reference: https://arxiv.org/abs/1904.09728

Sample usage:
```bash
bench eval social_iqa --model groq/llama-3.1-70b-versatile
```
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a Social IQA record to an OpenBench MCQSample.

    The dataset typically contains:
    - context: Description of a social situation
    - question: Question about the situation
    - answerA, answerB, answerC: Three answer options
    - label: Index (1-3) of correct answer
    """
    context = record.get("context", "")
    question = record.get("question", "")

    # Combine context and question
    if context:
        full_question = f"{context}\n\n{question}"
    else:
        full_question = question

    # Get options (different datasets may have different field names)
    options = []
    for key in ["answerA", "answer1", "choice1"]:
        if key in record:
            options.append(record[key])
            break

    for key in ["answerB", "answer2", "choice2"]:
        if key in record:
            options.append(record[key])
            break

    for key in ["answerC", "answer3", "choice3"]:
        if key in record:
            options.append(record[key])
            break

    # Fallback: try to extract from answers list
    if not options and "answers" in record:
        options = record["answers"]

    prompt_text = create_dynamic_multiple_choice_prompt(full_question, options)

    # Convert label to letter
    # Dataset label is 1-indexed (1,2,3)
    label = record.get("label", record.get("correct", 0))
    target = chr(64 + int(label))  # 1->A, 2->B, 3->C

    return MCQSample(
        input=prompt_text,
        target=target,
    )


@task
def social_iqa(split: str = "validation") -> Task:
    """Evaluate the Social IQA benchmark for social intelligence reasoning."""
    return MCQEval(
        name="social_iqa",
        dataset_path="social_i_qa",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        dataset_kwargs={"revision": "refs/convert/parquet"},
    )
