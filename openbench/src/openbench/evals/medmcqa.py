"""
MedMCQA: Medical Multiple Choice Question Answering Dataset

MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address
real-world medical entrance exam questions. It has more than 194k high-quality AIIMS & NEET PG
entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects.

Sample usage:
```bash
bench eval medmcqa --model "openrouter/openai/gpt-oss-120b" -M only=groq
```

Citation:
@inproceedings{pal2022medmcqa,
    title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
    author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
    booktitle={Conference on Health, Inference, and Learning},
    pages={248--260},
    year={2022},
    organization={PMLR}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a MedMCQA record to an OpenBench MCQSample.

    MedMCQA provides medical exam questions with 4 answer options.
    The dataset includes:
    - question: Question text
    - opa, opb, opc, opd: Four answer options
    - cop: Correct option index (0-3)
    - subject_name: Medical subject (e.g., Anatomy, Pharmacology)
    - exp: Explanation (optional)
    """
    question = record["question"]

    # Get the four options
    option_a = record["opa"]
    option_b = record["opb"]
    option_c = record["opc"]
    option_d = record["opd"]

    # Create prompt using standard 4-option template
    prompt = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        prompt=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    # Convert correct option index to letter (0->A, 1->B, 2->C, 3->D)
    correct_index = record["cop"]
    target = chr(65 + correct_index)

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "subject": record.get("subject_name", "unknown"),
            "topic": record.get("topic_name", "unknown"),
            "choice_type": record.get("choice_type", "unknown"),
        },
    )


@task
def medmcqa(split: str = "validation") -> Task:
    """
    Evaluate the MedMCQA dataset.

    MedMCQA contains medical entrance exam questions from AIIMS and NEET PG exams.
    The dataset tests knowledge across 21 medical subjects and 2400+ topics.

    Args:
        split: Dataset split to use (default: "validation")
               Options: "train" (182k), "validation" (4k), "test" (6k)

    Sample usage:
    ```bash
    bench eval medmcqa --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 100
    ```
    """
    return MCQEval(
        name="medmcqa",
        dataset_path="medmcqa",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        group_keys=["subject"],  # Group by medical subject
    )
