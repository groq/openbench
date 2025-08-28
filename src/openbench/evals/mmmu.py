"""MMMU (Massive Multi-discipline Multimodal Understanding) evaluation."""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mmmu import get_dataset
from openbench.scorers.mmmu import mmmu_scorer
from typing import Optional


@task
def mmmu(
    subset: Optional[str] = None,
    split: str = "validation",
    num_examples: Optional[int] = None,
) -> Task:
    """MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark.

    MMMU includes 11.5K meticulously collected multimodal questions from college exams,
    quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science,
    Health & Medicine, Humanities & Social Science, and Tech & Engineering across 30 subjects.

    Based on the paper: "MMMU: A Massive Multi-discipline Multimodal Understanding and
    Reasoning Benchmark for Expert AGI" by Yue et al. (2024).

    Args:
        subset: Optional subject subset (e.g., "Art", "Biology", "Physics").
                If None, uses all subjects
        split: Dataset split to use ("dev", "validation", "test")
        num_examples: Optional limit on number of examples to evaluate

    Returns:
        Task configured for MMMU evaluation
    """
    dataset = get_dataset(subset=subset, split=split, num_examples=num_examples)

    task_name = f"mmmu_{subset.lower()}" if subset else "mmmu"

    return Task(
        dataset=dataset,
        solver=[
            generate(),
        ],
        scorer=mmmu_scorer(),
        name=task_name,
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=1024,
        ),
    )


# Create individual subject tasks for the most common subjects
@task
def mmmu_art() -> Task:
    """MMMU Art subset evaluation."""
    return mmmu(subset="Art")


@task
def mmmu_biology() -> Task:
    """MMMU Biology subset evaluation."""
    return mmmu(subset="Biology")


@task
def mmmu_chemistry() -> Task:
    """MMMU Chemistry subset evaluation."""
    return mmmu(subset="Chemistry")


@task
def mmmu_computer_science() -> Task:
    """MMMU Computer Science subset evaluation."""
    return mmmu(subset="Computer_Science")


@task
def mmmu_economics() -> Task:
    """MMMU Economics subset evaluation."""
    return mmmu(subset="Economics")


@task
def mmmu_math() -> Task:
    """MMMU Math subset evaluation."""
    return mmmu(subset="Math")


@task
def mmmu_physics() -> Task:
    """MMMU Physics subset evaluation."""
    return mmmu(subset="Physics")


@task
def mmmu_psychology() -> Task:
    """MMMU Psychology subset evaluation."""
    return mmmu(subset="Psychology")


# Additional MMMU subject-specific tasks
@task
def mmmu_accounting() -> Task:
    """MMMU Accounting subset evaluation."""
    return mmmu(subset="Accounting")


@task
def mmmu_agriculture() -> Task:
    """MMMU Agriculture subset evaluation."""
    return mmmu(subset="Agriculture")


@task
def mmmu_architecture_and_engineering() -> Task:
    """MMMU Architecture and Engineering subset evaluation."""
    return mmmu(subset="Architecture_and_Engineering")


@task
def mmmu_art_theory() -> Task:
    """MMMU Art Theory subset evaluation."""
    return mmmu(subset="Art_Theory")


@task
def mmmu_basic_medical_science() -> Task:
    """MMMU Basic Medical Science subset evaluation."""
    return mmmu(subset="Basic_Medical_Science")


@task
def mmmu_clinical_medicine() -> Task:
    """MMMU Clinical Medicine subset evaluation."""
    return mmmu(subset="Clinical_Medicine")


@task
def mmmu_design() -> Task:
    """MMMU Design subset evaluation."""
    return mmmu(subset="Design")


@task
def mmmu_diagnostics_and_laboratory_medicine() -> Task:
    """MMMU Diagnostics and Laboratory Medicine subset evaluation."""
    return mmmu(subset="Diagnostics_and_Laboratory_Medicine")


@task
def mmmu_electronics() -> Task:
    """MMMU Electronics subset evaluation."""
    return mmmu(subset="Electronics")


@task
def mmmu_energy_and_power() -> Task:
    """MMMU Energy and Power subset evaluation."""
    return mmmu(subset="Energy_and_Power")


@task
def mmmu_finance() -> Task:
    """MMMU Finance subset evaluation."""
    return mmmu(subset="Finance")


@task
def mmmu_geography() -> Task:
    """MMMU Geography subset evaluation."""
    return mmmu(subset="Geography")


@task
def mmmu_history() -> Task:
    """MMMU History subset evaluation."""
    return mmmu(subset="History")


@task
def mmmu_literature() -> Task:
    """MMMU Literature subset evaluation."""
    return mmmu(subset="Literature")


@task
def mmmu_manage() -> Task:
    """MMMU Management subset evaluation."""
    return mmmu(subset="Manage")


@task
def mmmu_marketing() -> Task:
    """MMMU Marketing subset evaluation."""
    return mmmu(subset="Marketing")


@task
def mmmu_materials() -> Task:
    """MMMU Materials subset evaluation."""
    return mmmu(subset="Materials")


@task
def mmmu_mechanical_engineering() -> Task:
    """MMMU Mechanical Engineering subset evaluation."""
    return mmmu(subset="Mechanical_Engineering")


@task
def mmmu_music() -> Task:
    """MMMU Music subset evaluation."""
    return mmmu(subset="Music")


@task
def mmmu_pharmacy() -> Task:
    """MMMU Pharmacy subset evaluation."""
    return mmmu(subset="Pharmacy")


@task
def mmmu_public_health() -> Task:
    """MMMU Public Health subset evaluation."""
    return mmmu(subset="Public_Health")


@task
def mmmu_sociology() -> Task:
    """MMMU Sociology subset evaluation."""
    return mmmu(subset="Sociology")
