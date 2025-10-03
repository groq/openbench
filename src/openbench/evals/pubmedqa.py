"""
PubMedQA: A Dataset for Biomedical Research Question Answering

PubMedQA is a biomedical question answering dataset collected from PubMed abstracts.
The task is to answer research questions with yes/no/maybe based on the context from corresponding abstracts.

Sample usage:
```bash
bench eval pubmedqa --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@inproceedings{jin2019pubmedqa,
    title={PubMedQA: A Dataset for Biomedical Research Question Answering},
    author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William W and Lu, Xinghua},
    booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    pages={2567--2577},
    year={2019}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a PubMedQA record to an OpenBench MCQSample.

    PubMedQA provides biomedical research questions with context from abstracts.
    The dataset includes:
    - question: Research question
    - context: Dict with 'contexts' (list of abstract sections) and 'labels' (section types)
    - final_decision: Answer (yes/no/maybe)
    - long_answer: Detailed explanation
    """
    question = record["question"]

    # Get context sections
    context_data = record["context"]
    if isinstance(context_data, dict) and "contexts" in context_data:
        contexts = context_data["contexts"]
        # Join all context sections into one block
        context_text = "\n\n".join(contexts)
    else:
        context_text = str(context_data)

    # Create the full question with context
    full_question = f"{context_text}\n\nQuestion: {question}"

    # The answer is yes/no/maybe - convert to MCQ format
    options = ["yes", "no", "maybe"]

    # Create prompt
    prompt = create_dynamic_multiple_choice_prompt(full_question, options)

    # Get correct answer and map to letter
    answer = record["final_decision"].lower()
    answer_map = {
        "yes": "A",
        "no": "B",
        "maybe": "C",
    }

    if answer not in answer_map:
        raise ValueError(f"Unexpected answer: {answer}")

    target = answer_map[answer]

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "pubid": record.get("pubid"),
        },
    )


@task
def pubmedqa(split: str = "train") -> Task:
    """
    Evaluate the PubMedQA dataset.

    PubMedQA tests the ability to answer biomedical research questions based on
    abstract context with yes/no/maybe answers.

    Args:
        split: Dataset split to use (default: "train")
               Note: The pqa_labeled subset only has a "train" split (1000 samples)

    Sample usage:
    ```bash
    bench eval pubmedqa --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 100
    ```
    """
    return MCQEval(
        name="pubmedqa",
        dataset_path="pubmed_qa",
        subset_name="pqa_labeled",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
    )
