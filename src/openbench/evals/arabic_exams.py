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
    """Arabic Exams: Islamic Studies"""
    return arabic_exams(subset="Islamic Studies")


@task
def arabic_exams_math_high_school() -> Task:
    """Arabic Exams: Math (High School)"""
    return arabic_exams(subset="Math (High School)")


@task
def arabic_exams_math_primary_school() -> Task:
    """Arabic Exams: Math (Primary School)"""
    return arabic_exams(subset="Math (Primary School)")


@task
def arabic_exams_physics_high_school() -> Task:
    """Arabic Exams: Physics (High School)"""
    return arabic_exams(subset="Physics (High School)")


# Arabic Language (additional subsets)
@task
def arabic_exams_arabic_language_grammar() -> Task:
    """Arabic Exams: Arabic Language (Grammar)"""
    return arabic_exams(subset="Arabic Language (Grammar)")


@task
def arabic_exams_arabic_language_high_school() -> Task:
    """Arabic Exams: Arabic Language (High School)"""
    return arabic_exams(subset="Arabic Language (High School)")


@task
def arabic_exams_arabic_language_middle_school() -> Task:
    """Arabic Exams: Arabic Language (Middle School)"""
    return arabic_exams(subset="Arabic Language (Middle School)")


@task
def arabic_exams_arabic_language_primary_school() -> Task:
    """Arabic Exams: Arabic Language (Primary School)"""
    return arabic_exams(subset="Arabic Language (Primary School)")


# Biology
@task
def arabic_exams_biology_high_school() -> Task:
    """Arabic Exams: Biology (High School)"""
    return arabic_exams(subset="Biology (High School)")


# Civics
@task
def arabic_exams_civics_high_school() -> Task:
    """Arabic Exams: Civics (High School)"""
    return arabic_exams(subset="Civics (High School)")


@task
def arabic_exams_civics_middle_school() -> Task:
    """Arabic Exams: Civics (Middle School)"""
    return arabic_exams(subset="Civics (Middle School)")


# Computer Science (additional subsets)
@task
def arabic_exams_computer_science_middle_school() -> Task:
    """Arabic Exams: Computer Science (Middle School)"""
    return arabic_exams(subset="Computer Science (Middle School)")


@task
def arabic_exams_computer_science_primary_school() -> Task:
    """Arabic Exams: Computer Science (Primary School)"""
    return arabic_exams(subset="Computer Science (Primary School)")


# Driving Test
@task
def arabic_exams_driving_test() -> Task:
    """Arabic Exams: Driving Test"""
    return arabic_exams(subset="Driving Test")


# Economics
@task
def arabic_exams_economics_high_school() -> Task:
    """Arabic Exams: Economics (High School)"""
    return arabic_exams(subset="Economics (High School)")


@task
def arabic_exams_economics_middle_school() -> Task:
    """Arabic Exams: Economics (Middle School)"""
    return arabic_exams(subset="Economics (Middle School)")


@task
def arabic_exams_economics_university() -> Task:
    """Arabic Exams: Economics (University)"""
    return arabic_exams(subset="Economics (University)")


# General Knowledge
@task
def arabic_exams_general_knowledge() -> Task:
    """Arabic Exams: General Knowledge"""
    return arabic_exams(subset="General Knowledge")


@task
def arabic_exams_general_knowledge_middle_school() -> Task:
    """Arabic Exams: General Knowledge (Middle School)"""
    return arabic_exams(subset="General Knowledge (Middle School)")


@task
def arabic_exams_general_knowledge_primary_school() -> Task:
    """Arabic Exams: General Knowledge (Primary School)"""
    return arabic_exams(subset="General Knowledge (Primary School)")


# Geography
@task
def arabic_exams_geography_high_school() -> Task:
    """Arabic Exams: Geography (High School)"""
    return arabic_exams(subset="Geography (High School)")


@task
def arabic_exams_geography_middle_school() -> Task:
    """Arabic Exams: Geography (Middle School)"""
    return arabic_exams(subset="Geography (Middle School)")


@task
def arabic_exams_geography_primary_school() -> Task:
    """Arabic Exams: Geography (Primary School)"""
    return arabic_exams(subset="Geography (Primary School)")


# History
@task
def arabic_exams_history_high_school() -> Task:
    """Arabic Exams: History (High School)"""
    return arabic_exams(subset="History (High School)")


@task
def arabic_exams_history_middle_school() -> Task:
    """Arabic Exams: History (Middle School)"""
    return arabic_exams(subset="History (Middle School)")


@task
def arabic_exams_history_primary_school() -> Task:
    """Arabic Exams: History (Primary School)"""
    return arabic_exams(subset="History (Primary School)")


# Islamic Studies (additional subsets)
@task
def arabic_exams_islamic_studies_high_school() -> Task:
    """Arabic Exams: Islamic Studies (High School)"""
    return arabic_exams(subset="Islamic Studies (High School)")


@task
def arabic_exams_islamic_studies_middle_school() -> Task:
    """Arabic Exams: Islamic Studies (Middle School)"""
    return arabic_exams(subset="Islamic Studies (Middle School)")


@task
def arabic_exams_islamic_studies_primary_school() -> Task:
    """Arabic Exams: Islamic Studies (Primary School)"""
    return arabic_exams(subset="Islamic Studies (Primary School)")


# Law
@task
def arabic_exams_law_professional() -> Task:
    """Arabic Exams: Law (Professional)"""
    return arabic_exams(subset="Law (Professional)")


# Management
@task
def arabic_exams_management_university() -> Task:
    """Arabic Exams: Management (University)"""
    return arabic_exams(subset="Management (University)")


# Natural Science
@task
def arabic_exams_natural_science_middle_school() -> Task:
    """Arabic Exams: Natural Science (Middle School)"""
    return arabic_exams(subset="Natural Science (Middle School)")


@task
def arabic_exams_natural_science_primary_school() -> Task:
    """Arabic Exams: Natural Science (Primary School)"""
    return arabic_exams(subset="Natural Science (Primary School)")


# Philosophy
@task
def arabic_exams_philosophy_high_school() -> Task:
    """Arabic Exams: Philosophy (High School)"""
    return arabic_exams(subset="Philosophy (High School)")


# Political Science
@task
def arabic_exams_political_science_university() -> Task:
    """Arabic Exams: Political Science (University)"""
    return arabic_exams(subset="Political Science (University)")


# Social Science
@task
def arabic_exams_social_science_middle_school() -> Task:
    """Arabic Exams: Social Science (Middle School)"""
    return arabic_exams(subset="Social Science (Middle School)")


@task
def arabic_exams_social_science_primary_school() -> Task:
    """Arabic Exams: Social Science (Primary School)"""
    return arabic_exams(subset="Social Science (Primary School)")
