"""
BBQ: Bias Benchmark for QA

This benchmark evaluates social biases in question-answering systems across 11
demographic categories. BBQ tests whether models rely on stereotypes by presenting
questions in both ambiguous and disambiguated contexts, with questions targeting
stereotyped groups ("neg" polarity) and non-targeted groups ("nonneg" polarity).

Dataset: heegyu/bbq (HuggingFace)
Bias Categories: Age, Disability_status, Gender_identity, Nationality,
    Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender,
    Religion, SES, Sexual_orientation

Reference: https://arxiv.org/pdf/2110.08193

Sample usage:
```bash
# Run specific category
bench eval bbq_age --model groq/llama-3.3-70b-versatile

# Run all 11 categories
bench eval bbq_age bbq_disability_status bbq_gender_identity bbq_nationality bbq_physical_appearance bbq_race_ethnicity bbq_race_x_ses bbq_race_x_gender bbq_religion bbq_ses bbq_sexual_orientation --model groq/llama-3.3-70b-versatile

# Run with category parameter (legacy)
bench eval bbq -T category=Gender_identity --model groq/llama-3.3-70b-versatile
```
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a BBQ record to an OpenBench MCQSample.

    The dataset contains:
    - context: Narrative scenario describing a situation
    - question: Question about the context
    - ans0, ans1, ans2: Three answer choices
    - label: Correct answer index (0-2)
    - category: Bias category (e.g., "Age", "Gender_identity")
    - question_polarity: "neg" (targets stereotype) or "nonneg" (neutral)
    - context_condition: "ambig" (ambiguous) or "disambig" (clear context)
    """
    # Combine context and question
    full_question = f"{record['context']}\n\nQuestion: {record['question']}"

    # Create list of answer choices (3 options)
    choices = [
        record["ans0"],
        record["ans1"],
        record["ans2"],
    ]

    # Format as multiple choice prompt
    input_text = create_dynamic_multiple_choice_prompt(
        question=full_question, options=choices
    )

    # Map label (0-2) to MCQ letter (A-C)
    label_to_letter = {0: "A", 1: "B", 2: "C"}
    target = label_to_letter[record["label"]]

    return MCQSample(
        input=input_text,
        target=target,
        metadata={
            "category": record["category"],
            "question_polarity": record["question_polarity"],
            "context_condition": record["context_condition"],
        },
    )


# BBQ category mappings
BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]


@task
def bbq(category: str = "Age", split: str = "test") -> Task:
    """Evaluate a specific BBQ (Bias Benchmark for QA) category.

    Args:
        category: Bias category to evaluate. Valid options:
            - Age
            - Disability_status
            - Gender_identity
            - Nationality
            - Physical_appearance
            - Race_ethnicity
            - Race_x_SES
            - Race_x_gender
            - Religion
            - SES
            - Sexual_orientation
        split: Dataset split (default: "test")

    Returns:
        Task configured for BBQ evaluation with grouped metrics by category,
        question polarity, and context condition.
    """
    if category not in BBQ_CATEGORIES:
        available = ", ".join(BBQ_CATEGORIES)
        raise ValueError(f"Invalid BBQ category '{category}'. Available: {available}")

    task_name = f"bbq_{category.lower()}"

    return MCQEval(
        name=task_name,
        dataset_path="heegyu/bbq",
        subset_name=category,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        group_keys=["category", "question_polarity", "context_condition"],
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for bias evaluation
        ),
        dataset_kwargs={},
    )


# Individual task functions for each BBQ category
@task
def bbq_age(split: str = "test") -> Task:
    """BBQ: Age bias evaluation."""
    return bbq(category="Age", split=split)


@task
def bbq_disability_status(split: str = "test") -> Task:
    """BBQ: Disability status bias evaluation."""
    return bbq(category="Disability_status", split=split)


@task
def bbq_gender_identity(split: str = "test") -> Task:
    """BBQ: Gender identity bias evaluation."""
    return bbq(category="Gender_identity", split=split)


@task
def bbq_nationality(split: str = "test") -> Task:
    """BBQ: Nationality bias evaluation."""
    return bbq(category="Nationality", split=split)


@task
def bbq_physical_appearance(split: str = "test") -> Task:
    """BBQ: Physical appearance bias evaluation."""
    return bbq(category="Physical_appearance", split=split)


@task
def bbq_race_ethnicity(split: str = "test") -> Task:
    """BBQ: Race/ethnicity bias evaluation."""
    return bbq(category="Race_ethnicity", split=split)


@task
def bbq_race_x_ses(split: str = "test") -> Task:
    """BBQ: Race × Socioeconomic status bias evaluation."""
    return bbq(category="Race_x_SES", split=split)


@task
def bbq_race_x_gender(split: str = "test") -> Task:
    """BBQ: Race × Gender bias evaluation."""
    return bbq(category="Race_x_gender", split=split)


@task
def bbq_religion(split: str = "test") -> Task:
    """BBQ: Religion bias evaluation."""
    return bbq(category="Religion", split=split)


@task
def bbq_ses(split: str = "test") -> Task:
    """BBQ: Socioeconomic status bias evaluation."""
    return bbq(category="SES", split=split)


@task
def bbq_sexual_orientation(split: str = "test") -> Task:
    """BBQ: Sexual orientation bias evaluation."""
    return bbq(category="Sexual_orientation", split=split)
