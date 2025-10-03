"""
GLUE/SuperGLUE MCQ tasks for OpenBench.

This module implements the multiple-choice question (MCQ) tasks from the GLUE and SuperGLUE benchmarks:
- COPA (SuperGLUE): Choice of Plausible Alternatives - causal reasoning
- RTE (SuperGLUE): Recognizing Textual Entailment - natural language inference
- WiC (SuperGLUE): Word in Context - word sense disambiguation
- WSC (SuperGLUE): Winograd Schema Challenge - coreference resolution
- CB (SuperGLUE): CommitmentBank - natural language inference
- MultiRC (SuperGLUE): Multi-Sentence Reading Comprehension

Note: BoolQ is already implemented separately in boolq.py and is not included here.

Sample usage:
```bash
bench eval copa --model "groq/llama-3.1-8b-instant"
bench eval rte_superglue --model "groq/llama-3.1-8b-instant"
```

The prompts are based on the lighteval implementations:
https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/default_prompts.py
"""

from inspect_ai.solver._multiple_choice import prompt
from inspect_ai import Task, task
from inspect_ai.solver import Choices
from openbench.utils.mcq import MCQEval, MCQSample

# MCQ Template
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


# ==================== COPA ====================
def record_to_mcq_sample_copa(record: dict) -> MCQSample:
    """Convert a COPA record to an OpenBench MCQSample.

    COPA tests causal reasoning by asking which of two alternatives
    is more plausibly the cause or effect of a given premise.
    """
    connector = {"cause": "because", "effect": "therefore"}[record["question"]]
    premise = record["premise"].strip()
    if premise.endswith("."):
        premise = premise[:-1]

    # Format: "The man turned on the faucet therefore"
    input_question = f"{premise} {connector}"

    # Lowercase first letter of each choice
    choice1 = record["choice1"]
    choice2 = record["choice2"]
    choice1 = choice1[0].lower() + choice1[1:] if choice1 else choice1
    choice2 = choice2[0].lower() + choice2[1:] if choice2 else choice2

    input_msg = prompt(
        question=input_question,
        choices=Choices([choice1, choice2]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    int_to_char = {0: "A", 1: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# ==================== RTE ====================
def record_to_mcq_sample_rte(record: dict) -> MCQSample:
    """Convert an RTE record to an OpenBench MCQSample.

    RTE (Recognizing Textual Entailment) tests whether a hypothesis
    can be inferred from a premise.
    """
    input_question = (
        f"{record['premise']}\n"
        f"Question: {record['hypothesis']} True or False?\n"
        f"Answer:"
    )

    input_msg = prompt(
        question=input_question,
        choices=Choices(["True", "False"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 0 = entailment (True), Label 1 = not_entailment (False)
    int_to_char = {0: "A", 1: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# ==================== WiC ====================
def record_to_mcq_sample_wic(record: dict) -> MCQSample:
    """Convert a WiC record to an OpenBench MCQSample.

    WiC (Word in Context) tests whether a word is used with the same
    meaning in two different sentences.
    """
    input_question = (
        f"Sentence 1: {record['sentence1']}\n"
        f"Sentence 2: {record['sentence2']}\n"
        f"Question: Is the word '{record['word']}' used in the same way in the two sentences above?\n"
        f"Answer:"
    )

    input_msg = prompt(
        question=input_question,
        choices=Choices(["no", "yes"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 0 = different meaning (no), Label 1 = same meaning (yes)
    int_to_char = {0: "A", 1: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# ==================== WSC ====================
def record_to_mcq_sample_wsc(record: dict) -> MCQSample:
    """Convert a WSC record to an OpenBench MCQSample.

    WSC (Winograd Schema Challenge) tests coreference resolution by
    determining whether a pronoun refers to a specific noun.
    """
    input_question = (
        f"Passage: {record['text']}\n"
        f"Question: In the passage above, does the pronoun '{record['span2_text']}' "
        f"refer to '{record['span1_text']}'?\n"
        f"Answer:"
    )

    input_msg = prompt(
        question=input_question,
        choices=Choices(["no", "yes"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 0 = does not refer (no), Label 1 = refers (yes)
    int_to_char = {0: "A", 1: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# ==================== CB ====================
def record_to_mcq_sample_cb(record: dict) -> MCQSample:
    """Convert a CB record to an OpenBench MCQSample.

    CB (CommitmentBank) tests natural language inference with three labels:
    entailment, contradiction, and neutral.
    """
    input_question = (
        f"{record['premise']}\n"
        f"Question: {record['hypothesis']}. True, False or Neither?\n"
        f"Answer:"
    )

    input_msg = prompt(
        question=input_question,
        choices=Choices(["True", "False", "Neither"]),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 0 = entailment (True), Label 1 = contradiction (False), Label 2 = neutral (Neither)
    int_to_char = {0: "A", 1: "B", 2: "C"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# ==================== MultiRC ====================
def record_to_mcq_sample_multirc(record: dict) -> MCQSample:
    """Convert a MultiRC record to an OpenBench MCQSample.

    MultiRC (Multi-Sentence Reading Comprehension) tests whether
    a given answer to a question is correct based on a paragraph.
    """
    input_question = (
        f"{record['paragraph']}\n" f"Question: {record['question']}\n" f"Answer:"
    )

    # Each sample asks "Is this answer correct?"
    input_msg = prompt(
        question=input_question,
        choices=Choices(
            [
                f"{record['answer']}\nIs the answer correct? yes",
                f"{record['answer']}\nIs the answer correct? no",
            ]
        ),
        template=str(SINGLE_ANSWER_TEMPLATE),
    )

    # Label 1 = correct (yes), Label 0 = incorrect (no)
    # So if label=1, answer is A (yes), if label=0, answer is B (no)
    int_to_char = {1: "A", 0: "B"}
    return MCQSample(
        input=input_msg,
        target=int_to_char[record["label"]],
    )


# Mapping of subset names to their configuration
SUPERGLUE_SUBSET_CONFIG = {
    "copa": {
        "task_name": "copa",
        "dataset_subset": "copa",
        "converter": record_to_mcq_sample_copa,
    },
    "rte": {
        "task_name": "rte_superglue",
        "dataset_subset": "rte",
        "converter": record_to_mcq_sample_rte,
    },
    "wic": {
        "task_name": "wic",
        "dataset_subset": "wic",
        "converter": record_to_mcq_sample_wic,
    },
    "wsc": {
        "task_name": "wsc",
        "dataset_subset": "wsc",
        "converter": record_to_mcq_sample_wsc,
    },
    "cb": {
        "task_name": "cb",
        "dataset_subset": "cb",
        "converter": record_to_mcq_sample_cb,
    },
    "multirc": {
        "task_name": "multirc",
        "dataset_subset": "multirc",
        "converter": record_to_mcq_sample_multirc,
    },
}


# Main SuperGLUE function - all subset tasks call this
@task
def superglue(subset: str = "copa", split: str = "validation") -> Task:
    """
    Family benchmark for SuperGLUE - run any SuperGLUE subset by name.

    Args:
        subset: SuperGLUE subset to evaluate. Available subsets:
                - copa: Choice of Plausible Alternatives (causal reasoning)
                - rte: Recognizing Textual Entailment
                - wic: Word in Context (word sense disambiguation)
                - wsc: Winograd Schema Challenge (coreference resolution)
                - cb: CommitmentBank (3-way NLI)
                - multirc: Multi-Sentence Reading Comprehension
        split: Dataset split to use (default: "validation")

    Returns:
        Task: The specified SuperGLUE subset task
    """
    if subset not in SUPERGLUE_SUBSET_CONFIG:
        available = ", ".join(SUPERGLUE_SUBSET_CONFIG.keys())
        raise ValueError(f"Invalid SuperGLUE subset '{subset}'. Available: {available}")

    config = SUPERGLUE_SUBSET_CONFIG[subset]

    return MCQEval(
        name=str(config["task_name"]),
        dataset_path="super_glue",
        subset_name=str(config["dataset_subset"]),
        record_to_mcq_sample=config["converter"],  # type: ignore[arg-type]
        split=split,
    )


# Individual task functions - convenience wrappers that call superglue(subset=...)
@task
def copa(split: str = "validation") -> Task:
    """
    COPA: Choice of Plausible Alternatives (SuperGLUE)

    Tests causal reasoning by selecting the more plausible cause or effect
    from two alternatives given a premise.
    """
    return superglue(subset="copa", split=split)


@task
def rte_superglue(split: str = "validation") -> Task:
    """
    RTE: Recognizing Textual Entailment (SuperGLUE)

    Tests natural language inference by determining if a hypothesis
    can be inferred from a given premise.
    """
    return superglue(subset="rte", split=split)


@task
def wic(split: str = "validation") -> Task:
    """
    WiC: Word in Context (SuperGLUE)

    Tests word sense disambiguation by determining if a word has the
    same meaning in two different sentences.
    """
    return superglue(subset="wic", split=split)


@task
def wsc(split: str = "validation") -> Task:
    """
    WSC: Winograd Schema Challenge (SuperGLUE)

    Tests coreference resolution by determining whether a pronoun
    refers to a specific noun phrase in a passage.
    """
    return superglue(subset="wsc", split=split)


@task
def cb(split: str = "validation") -> Task:
    """
    CB: CommitmentBank (SuperGLUE)

    Tests natural language inference with three-way classification:
    entailment, contradiction, or neutral.
    """
    return superglue(subset="cb", split=split)


@task
def multirc(split: str = "validation") -> Task:
    """
    MultiRC: Multi-Sentence Reading Comprehension (SuperGLUE)

    Tests reading comprehension by determining whether proposed answers
    to questions about a paragraph are correct.
    """
    return superglue(subset="multirc", split=split)
