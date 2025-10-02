from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert an Arabic MMLU/Exams record to an OpenBench MCQSample."""
    # Build options list (can have 4 or 5 options)
    # Note: Dataset uses spaces in column names (e.g., "Option 1" not "Option_1")
    options = [
        record["Option 1"],
        record["Option 2"],
        record["Option 3"],
        record["Option 4"],
    ]

    # Add Option 5 if it exists and is not None/empty
    if record.get("Option 5") and str(record.get("Option 5")).strip():
        options.append(record["Option 5"])

    # Create dynamic prompt with variable number of options
    question_text = record["Question"]
    if record.get("Context") and str(record.get("Context")).strip():
        question_text = f"{record['Context']}\n\n{question_text}"

    prompt = create_dynamic_multiple_choice_prompt(question_text, options)

    # Answer Key should already be a letter (A, B, C, D, or E)
    return MCQSample(
        input=prompt,
        target=record["Answer Key"],
        metadata={
            "subject": record.get("Subject", "general"),
            "group": record.get("Group", "general"),
            "level": record.get("Level", "general"),
            "source": record.get("Source", "unknown"),
            "country": record.get("Country", "unknown"),
        },
    )


@task
def arabic_exams(subset: str = "Accounting (University)") -> Task:
    """
    Evaluate Arabic MMLU dataset from MBZUAI/ArabicMMLU.

    The first multi-task language understanding benchmark for Arabic language,
    sourced from school exams across diverse educational levels in different
    countries spanning North Africa, the Levant, and the Gulf regions.

    Comprises 40 tasks and 14,575 multiple-choice questions in Modern Standard Arabic (MSA).

    Args:
        subset: Subject to evaluate (default: "Accounting (University)")
    """
    return MCQEval(
        name=f"arabic_exams_{subset.replace(' ', '_').replace('(', '').replace(')', '').lower()}",
        dataset_path="MBZUAI/ArabicMMLU",
        subset_name=subset,
        record_to_mcq_sample=record_to_mcq_sample,
        split="test",
        auto_id=True,
        config=GenerateConfig(
            temperature=0.5,
        ),
        group_keys=["group", "level"],
    )


# Create wrapper functions for common subjects
@task
def arabic_exams_accounting_university() -> Task:
    """Arabic Exams: Accounting (University)"""
    return arabic_exams(subset="Accounting (University)")


@task
def arabic_exams_arabic_language_general() -> Task:
    """Arabic Exams: Arabic Language (General)"""
    return arabic_exams(subset="Arabic Language (General)")


@task
def arabic_exams_computer_science_high_school() -> Task:
    """Arabic Exams: Computer Science (High School)"""
    return arabic_exams(subset="Computer Science (High School)")


@task
def arabic_exams_computer_science_university() -> Task:
    """Arabic Exams: Computer Science (University)"""
    return arabic_exams(subset="Computer Science (University)")


@task
def arabic_exams_islamic_studies_general() -> Task:
    """Arabic Exams: Islamic Studies (General)"""
    return arabic_exams(subset="Islamic Studies (General)")


@task
def arabic_exams_math_high_school() -> Task:
    """Arabic Exams: Math (High School)"""
    return arabic_exams(subset="Math (High School)")


@task
def arabic_exams_physics_high_school() -> Task:
    """Arabic Exams: Physics (High School)"""
    return arabic_exams(subset="Physics (High School)")


@task
def arabic_exams_physics_university() -> Task:
    """Arabic Exams: Physics (University)"""
    return arabic_exams(subset="Physics (University)")
