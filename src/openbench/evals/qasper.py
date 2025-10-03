"""
QASPER (Question Answering on Scientific Papers) - Binary MCQ Variant

QASPER is a dataset for information-seeking question answering on scientific
research papers. This implementation focuses on the yes/no questions subset,
evaluating binary classification on paper abstracts.

Dataset: allenai/qasper
Split: validation (199 papers)
Format: Binary MCQ (yes/no)
Metric: Accuracy via loglikelihood

The dataset contains multiple questions per paper, but only yes/no questions
are included in this MCQ evaluation. Other question types (extractive,
abstractive, unanswerable) are filtered out.

Sample usage:
```bash
bench eval qasper_ll --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{Dasigi2021QASPER,
    title={A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
    author={Pradeep Dasigi and Kyle Lo and Iz Beltagy and Arman Cohan and Noah A. Smith and Matt Gardner},
    booktitle={NAACL},
    year={2021}
}
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Choices
from inspect_ai.solver._multiple_choice import prompt
from openbench.utils.mcq import MCQEval, MCQSample


# MCQ prompt template for binary yes/no questions
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


def extract_answer(answer_dict: dict) -> str | None:
    """
    Extract yes/no answer from QASPER answer dictionary.

    QASPER answers have multiple possible types:
    - free_form_answer: Open-ended text answer
    - extractive_spans: Spans from the paper
    - unanswerable: Question cannot be answered from paper
    - yes_no: Boolean yes/no question

    For qasper_ll (MCQ), we only care about yes_no questions.

    Returns:
        "yes" if yes_no is true and answer is yes
        "no" if yes_no is true and answer is no
        None if not a yes/no question
    """
    # Priority order from lighteval implementation
    keys_to_check = ["free_form_answer", "extractive_spans"]
    for key in keys_to_check:
        if answer_dict.get(key):
            # Not a yes/no question, skip
            return None

    if answer_dict.get("unanswerable"):
        # Not a yes/no question, skip
        return None

    if answer_dict.get("yes_no"):
        # This is a yes/no question
        # The yes_no field contains the boolean answer
        return "yes" if answer_dict["yes_no"] else "no"

    # Default to no if no answer type matches (from lighteval)
    return "no"


def record_to_mcq_sample_qasper(record: dict) -> list[MCQSample]:
    """
    Convert QASPER record to list of MCQSamples (yes/no questions only).

    QASPER records contain multiple questions per paper. Each question may have
    multiple answers. We extract only yes/no questions and create one MCQSample
    per yes/no question.

    Args:
        record: QASPER dataset record with keys:
            - title: Paper title
            - abstract: Paper abstract
            - qas: Dict with "question" (list) and "answers" (list of lists)

    Returns:
        List of MCQSample objects (can be empty if no yes/no questions)
    """
    title = record["title"]
    abstract = record["abstract"]

    # Extract questions and answers
    questions = record["qas"]["question"]
    answer_lists = record["qas"]["answers"]

    samples = []

    # Iterate through all questions and their answer lists
    for question, answer_list in zip(questions, answer_lists):
        # Each question has multiple annotator answers
        for answer_dict in answer_list["answer"]:
            # Extract the answer type
            gold = extract_answer(answer_dict)

            # Only include yes/no questions
            if gold in ["yes", "no"]:
                # Create the question context
                question_context = (
                    f"TITLE: {title}\nABSTRACT: {abstract}\n\nQ: {question}"
                )

                # Format as MCQ prompt with yes/no choices
                input_msg = prompt(
                    question=question_context,
                    choices=Choices(["yes", "no"]),
                    template=str(SINGLE_ANSWER_TEMPLATE),
                )

                # Map answer to target letter
                # "yes" → A (index 0), "no" → B (index 1)
                target_letter = "A" if gold == "yes" else "B"

                samples.append(
                    MCQSample(
                        input=input_msg,
                        target=target_letter,
                        metadata={
                            "answer_type": "yes_no",
                            "question": question[:100],  # Truncate for logging
                        },
                    )
                )

    return samples


@task
def qasper_ll(split: str = "validation") -> Task:
    """
    QASPER Log-Likelihood: Binary yes/no QA on scientific papers.

    Evaluates reading comprehension on scientific papers using binary (yes/no)
    multiple choice questions. Only questions with yes/no answers are included;
    other question types (extractive, abstractive, unanswerable) are filtered out.

    Args:
        split: Dataset split to use (default: "validation")
            - "validation": 199 papers (recommended)
            - "train": 2,593 papers

    Returns:
        Task configured for MCQ evaluation with binary choices

    Note:
        Each paper contains multiple questions, but only yes/no questions are
        evaluated. The number of samples will be less than the number of papers
        due to filtering.
    """
    valid_splits = ["train", "validation"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="qasper_ll",
        dataset_path="allenai/qasper",
        subset_name="qasper",
        record_to_mcq_sample=record_to_mcq_sample_qasper,
        split=split,
        auto_id=True,  # Auto-generate IDs since we flatten multiple samples per record
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for consistency
        ),
    )
