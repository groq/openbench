"""
QA4MRE - Question Answering for Machine Reading Evaluation

QA4MRE is a benchmark for evaluating machine reading comprehension. The task
involves reading a document and answering multiple-choice questions with 5 options.
This dataset was part of the CLEF (Conference and Labs of the Evaluation Forum)
shared tasks from 2011-2013.

This implementation includes three years of the English main task:
- QA4MRE 2011 (English)
- QA4MRE 2012 (English)
- QA4MRE 2013 (English)

Dataset: qa4mre
Paper: https://www.cs.cmu.edu/~hovy/papers/13HLT-QA4MRE.pdf

Sample usage:
```bash
bench eval qa4mre_2011 --model "openrouter/openai/gpt-oss-120b" -M only=groq
bench eval qa4mre_2012 --model "openrouter/openai/gpt-oss-120b" -M only=groq
bench eval qa4mre_2013 --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@inproceedings{sutcliffe2013qa4mre,
    title={Overview of QA4MRE at CLEF 2013: Question Answering for Machine Reading Evaluation},
    author={Sutcliffe, Richard and Hasan, Mahmoud and Pen{\'a}s, Anselmo and Forner, Pamela and Rodrigo, {\'A}lvaro and Osenova, Petya and Sang, Erik Tjong Kim and Derval, Manuel},
    booktitle={CLEF 2013},
    year={2013}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


# Mapping of subset names
QA4MRE_SUBSETS = {
    "2011": "2011.main.EN",
    "2012": "2012.main.EN",
    "2013": "2013.main.EN",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a QA4MRE record to an OpenBench MCQSample.

    Args:
        record: Dictionary containing:
            - document_str: The reading document
            - question_str: The question about the document
            - answer_options: Dictionary with lists:
                - answer_id: List of answer IDs
                - answer_str: List of answer strings
            - correct_answer_id: The ID of the correct answer

    Returns:
        MCQSample with formatted prompt and target answer
    """
    document = record["document_str"]
    question = record["question_str"]

    # Extract options from the answer_options dict
    # The structure is {'answer_id': ['1', '2', ...], 'answer_str': ['text1', 'text2', ...]}
    answer_options = record["answer_options"]
    answer_ids = answer_options["answer_id"]
    answer_strs = answer_options["answer_str"]

    # Create list of options
    options = answer_strs

    # Find the correct answer index
    correct_answer_id = record["correct_answer_id"]
    correct_index = answer_ids.index(correct_answer_id)
    target = chr(65 + correct_index)  # Convert 0->A, 1->B, etc.

    # Create the full prompt: document + question + options
    full_question = f"Document:\n{document}\n\nQuestion: {question}"
    prompt = create_dynamic_multiple_choice_prompt(full_question, options)

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "document_id": record.get("document_id", ""),
            "question_id": record.get("question_id", ""),
            "topic_name": record.get("topic_name", ""),
        },
    )


@task
def qa4mre(year: str = "2011", split: str = "train") -> Task:
    """
    Family benchmark for QA4MRE - run any year by name.

    Args:
        year: QA4MRE year to evaluate ("2011", "2012", "2013")
              Default: "2011"
        split: Dataset split to use (default: "train")
               Note: QA4MRE only has 'train' split available

    Returns:
        Task: Inspect AI task for QA4MRE evaluation
    """
    if year not in QA4MRE_SUBSETS:
        available = ", ".join(QA4MRE_SUBSETS.keys())
        raise ValueError(f"Invalid year '{year}'. Available: {available}")

    return MCQEval(
        name=f"qa4mre_{year}",
        dataset_path="qa4mre",
        subset_name=QA4MRE_SUBSETS[year],
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )


@task
def qa4mre_2011(split: str = "train") -> Task:
    """QA4MRE 2011 - Question Answering for Machine Reading Evaluation (English)"""
    return qa4mre(year="2011", split=split)


@task
def qa4mre_2012(split: str = "train") -> Task:
    """QA4MRE 2012 - Question Answering for Machine Reading Evaluation (English)"""
    return qa4mre(year="2012", split=split)


@task
def qa4mre_2013(split: str = "train") -> Task:
    """QA4MRE 2013 - Question Answering for Machine Reading Evaluation (English)"""
    return qa4mre(year="2013", split=split)
