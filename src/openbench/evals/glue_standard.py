"""
GLUE: General Language Understanding Evaluation

GLUE is a collection of resources for training, evaluating, and analyzing natural language
understanding systems. GLUE consists of 9 sentence or sentence-pair language understanding
tasks built on established existing datasets.

Tasks:
- CoLA: Corpus of Linguistic Acceptability (grammatical acceptability)
- SST-2: Stanford Sentiment Treebank (sentiment classification)
- MRPC: Microsoft Research Paraphrase Corpus (paraphrase detection)
- QQP: Quora Question Pairs (question similarity)
- STS-B: Semantic Textual Similarity Benchmark (sentence similarity)
- MNLI: Multi-Genre Natural Language Inference (textual entailment)
- QNLI: Question-answering Natural Language Inference (QA/NLI)
- RTE: Recognizing Textual Entailment (textual entailment)
- WNLI: Winograd Natural Language Inference (coreference/NLI)

Sample usage:
```bash
bench eval glue_cola --model "groq/llama-3.1-8b-instant"
bench eval glue_sst2 --model "groq/llama-3.1-8b-instant"
bench eval glue_mnli --model "groq/llama-3.1-8b-instant"
```

Citation:
@inproceedings{wang2018glue,
    title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
    author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
    booktitle={Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP},
    pages={353--355},
    year={2018}
}
"""

from inspect_ai import Task, task
from inspect_ai.solver import Choices
from inspect_ai.solver._multiple_choice import prompt
from openbench.utils.mcq import MCQEval, MCQSample

# Binary classification template
BINARY_TEMPLATE = r"""
Answer the following question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

# GLUE task configurations
GLUE_TASKS = {
    "cola": {"split": "validation", "num_choices": 2},
    "sst2": {"split": "validation", "num_choices": 2},
    "mrpc": {"split": "validation", "num_choices": 2},
    "qqp": {"split": "validation", "num_choices": 2},
    "stsb": {"split": "validation", "num_choices": 2},
    "mnli": {"split": "validation_matched", "num_choices": 3},
    "mnli_mismatched": {"split": "validation_mismatched", "num_choices": 3},
    "qnli": {"split": "validation", "num_choices": 2},
    "rte": {"split": "validation", "num_choices": 2},
    "wnli": {"split": "validation", "num_choices": 2},
}


def record_to_mcq_sample_cola(record: dict) -> MCQSample:
    """Convert CoLA record to MCQSample."""
    sentence = record["sentence"]
    label = record["label"]  # 0 = unacceptable, 1 = acceptable

    question = f"Is the following sentence grammatically acceptable?\n\n{sentence}"
    choices_list = ["No (unacceptable)", "Yes (acceptable)"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_sst2(record: dict) -> MCQSample:
    """Convert SST-2 record to MCQSample."""
    sentence = record["sentence"]
    label = record["label"]  # 0 = negative, 1 = positive

    question = f"What is the sentiment of the following sentence?\n\n{sentence}"
    choices_list = ["Negative", "Positive"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_mrpc(record: dict) -> MCQSample:
    """Convert MRPC record to MCQSample."""
    sentence1 = record["sentence1"]
    sentence2 = record["sentence2"]
    label = record["label"]  # 0 = not paraphrase, 1 = paraphrase

    question = f"Are these two sentences paraphrases of each other?\n\nSentence 1: {sentence1}\n\nSentence 2: {sentence2}"
    choices_list = ["No", "Yes"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_qqp(record: dict) -> MCQSample:
    """Convert QQP record to MCQSample."""
    question1 = record["question1"]
    question2 = record["question2"]
    label = record["label"]  # 0 = not duplicate, 1 = duplicate

    question = f"Are these two questions asking the same thing?\n\nQuestion 1: {question1}\n\nQuestion 2: {question2}"
    choices_list = ["No", "Yes"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_stsb(record: dict) -> MCQSample:
    """Convert STS-B record to MCQSample."""
    sentence1 = record["sentence1"]
    sentence2 = record["sentence2"]
    # STS-B has continuous similarity scores 0-5, we'll bin them
    score = record["label"]

    # Bin into low (0-2) and high (2.5-5) similarity
    label = 0 if score < 2.5 else 1

    question = f"How similar are these two sentences?\n\nSentence 1: {sentence1}\n\nSentence 2: {sentence2}"
    choices_list = ["Low similarity", "High similarity"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_mnli(record: dict) -> MCQSample:
    """Convert MNLI record to MCQSample."""
    premise = record["premise"]
    hypothesis = record["hypothesis"]
    label = record["label"]  # 0 = entailment, 1 = neutral, 2 = contradiction

    question = f"Given the premise, what is the relationship to the hypothesis?\n\nPremise: {premise}\n\nHypothesis: {hypothesis}"
    choices_list = ["Entailment", "Neutral", "Contradiction"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = chr(65 + label)  # 0->A, 1->B, 2->C

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_qnli(record: dict) -> MCQSample:
    """Convert QNLI record to MCQSample."""
    question = record["question"]
    sentence = record["sentence"]
    label = record["label"]  # 0 = entailment, 1 = not entailment

    question_text = f"Does the sentence answer the question?\n\nQuestion: {question}\n\nSentence: {sentence}"
    choices_list = ["Yes (entailment)", "No (not entailment)"]

    input_msg = prompt(
        question=question_text,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_rte(record: dict) -> MCQSample:
    """Convert RTE record to MCQSample."""
    sentence1 = record["sentence1"]
    sentence2 = record["sentence2"]
    label = record["label"]  # 0 = entailment, 1 = not entailment

    question = f"Does the first sentence entail the second?\n\nSentence 1: {sentence1}\n\nSentence 2: {sentence2}"
    choices_list = ["Yes (entailment)", "No (not entailment)"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


def record_to_mcq_sample_wnli(record: dict) -> MCQSample:
    """Convert WNLI record to MCQSample."""
    sentence1 = record["sentence1"]
    sentence2 = record["sentence2"]
    label = record["label"]  # 0 = not entailment, 1 = entailment

    question = f"Does the first sentence entail the second?\n\nSentence 1: {sentence1}\n\nSentence 2: {sentence2}"
    choices_list = ["No (not entailment)", "Yes (entailment)"]

    input_msg = prompt(
        question=question,
        choices=Choices(choices_list),
        template=BINARY_TEMPLATE,
    )

    target = "A" if label == 0 else "B"

    return MCQSample(input=input_msg, target=target, id=str(record.get("idx", "")))


# Converter mapping
GLUE_CONVERTERS = {
    "cola": record_to_mcq_sample_cola,
    "sst2": record_to_mcq_sample_sst2,
    "mrpc": record_to_mcq_sample_mrpc,
    "qqp": record_to_mcq_sample_qqp,
    "stsb": record_to_mcq_sample_stsb,
    "mnli": record_to_mcq_sample_mnli,
    "mnli_mismatched": record_to_mcq_sample_mnli,
    "qnli": record_to_mcq_sample_qnli,
    "rte": record_to_mcq_sample_rte,
    "wnli": record_to_mcq_sample_wnli,
}


@task
def glue(task_name: str = "cola", split: str | None = None) -> Task:
    """
    Family benchmark for GLUE tasks.

    Args:
        task_name: GLUE task to evaluate (default: "cola")
        split: Dataset split to use (default: task-specific)

    Returns:
        Task: The specified GLUE task
    """
    if task_name not in GLUE_TASKS:
        available = ", ".join(GLUE_TASKS.keys())
        raise ValueError(f"Invalid GLUE task '{task_name}'. Available: {available}")

    config = GLUE_TASKS[task_name]
    if split is None:
        split = str(config["split"])

    # Handle MNLI naming
    subset_name = "mnli" if task_name == "mnli_mismatched" else task_name

    return MCQEval(
        name=f"glue_{task_name}",
        dataset_path="glue",
        subset_name=subset_name,
        record_to_mcq_sample=GLUE_CONVERTERS[task_name],
        split=split,
    )


# Individual task wrapper functions
@task
def glue_cola(split: str = "validation") -> Task:
    """GLUE: CoLA - Corpus of Linguistic Acceptability"""
    return glue(task_name="cola", split=split)


@task
def glue_sst2(split: str = "validation") -> Task:
    """GLUE: SST-2 - Stanford Sentiment Treebank"""
    return glue(task_name="sst2", split=split)


@task
def glue_mrpc(split: str = "validation") -> Task:
    """GLUE: MRPC - Microsoft Research Paraphrase Corpus"""
    return glue(task_name="mrpc", split=split)


@task
def glue_qqp(split: str = "validation") -> Task:
    """GLUE: QQP - Quora Question Pairs"""
    return glue(task_name="qqp", split=split)


@task
def glue_stsb(split: str = "validation") -> Task:
    """GLUE: STS-B - Semantic Textual Similarity Benchmark"""
    return glue(task_name="stsb", split=split)


@task
def glue_mnli(split: str = "validation_matched") -> Task:
    """GLUE: MNLI - Multi-Genre Natural Language Inference"""
    return glue(task_name="mnli", split=split)


@task
def glue_mnli_mismatched(split: str = "validation_mismatched") -> Task:
    """GLUE: MNLI Mismatched - Multi-Genre Natural Language Inference"""
    return glue(task_name="mnli_mismatched", split=split)


@task
def glue_qnli(split: str = "validation") -> Task:
    """GLUE: QNLI - Question-answering Natural Language Inference"""
    return glue(task_name="qnli", split=split)


@task
def glue_rte(split: str = "validation") -> Task:
    """GLUE: RTE - Recognizing Textual Entailment"""
    return glue(task_name="rte", split=split)


@task
def glue_wnli(split: str = "validation") -> Task:
    """GLUE: WNLI - Winograd Natural Language Inference"""
    return glue(task_name="wnli", split=split)
