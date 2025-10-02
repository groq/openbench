"""openbench implementation of SuperGPQA.
SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines

Implemented by Aarush Sah
"""

from inspect_ai import task
from openbench.utils.mcq import MCQSample, MCQEval
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record) -> MCQSample:
    """Convert a SuperGPQA record to an Inspect Sample."""
    question = record["question"]
    choices = record["options"]
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    # Create metadata dict with all extra fields
    metadata = {
        "uuid": record["uuid"],
        "discipline": record["discipline"],
        "field": record["field"],
        "subfield": record["subfield"],
        "difficulty": record["difficulty"],
        "is_calculation": record["is_calculation"],
        "answer_text": record["answer"],  # Store the full answer text
    }

    return MCQSample(
        input=prompt,
        target=record["answer_letter"],
        id=record["uuid"],
        metadata=metadata,
    )


@task
def supergpqa(
    field: str | None = None,
    subfield: str | None = None,
    difficulty: str | None = None,
    discipline: str | None = None,
):
    """SuperGPQA dataset task (MCQ Abstracted) with optional filtering.

    Filters supported: field, subfield, difficulty, discipline.
    """

    # Wrap the mapper to apply record-level filtering (return [] to drop non-matching records)
    def filtered_records_to_mcq_sample(record):
        if field and record.get("field") != field:
            return []
        if subfield and record.get("subfield") != subfield:
            return []
        if difficulty and record.get("difficulty") != difficulty:
            return []
        if discipline and record.get("discipline") != discipline:
            return []
        return record_to_mcq_sample(record)

    return MCQEval(
        name="supergpqa",
        dataset_path="m-a-p/SuperGPQA",
        record_to_mcq_sample=filtered_records_to_mcq_sample,
        split="train",  # Only train split is available
        group_keys=["difficulty"],
    )
