"""MMMU (Massive Multi-discipline Multimodal Understanding) evaluation."""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mmmu import get_dataset
from openbench.scorers.mcq import create_mcq_scorer
from openbench.scorers.mmmu import mmmu_mixed_scorer
from typing import Optional


@task
def mmmu(
    subset: Optional[str] = None,
    split: str = "validation",
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

    Returns:
        Task configured for MMMU evaluation
    """
    dataset = get_dataset(subset=subset, split=split)

    task_name = f"mmmu_{subset.lower()}" if subset else "mmmu"

    return Task(
        dataset=dataset,
        solver=[
            generate(),
        ],
        scorer=mmmu_mixed_scorer(group_keys=["subfield", "topic_difficulty"])(),
        name=task_name,
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )


# Dynamically create subject-specific tasks
def _create_subject_task(subject_name: str):
    """Create a task function for a specific MMMU subject."""

    @task
    def subject_task() -> Task:
        return mmmu(subset=subject_name)

    # Set a proper name for the function
    subject_task.__name__ = f"mmmu_{subject_name.lower()}"
    subject_task.__doc__ = (
        f"MMMU {subject_name.replace('_', ' ').title()} subset evaluation."
    )

    return subject_task


# Create all subject-specific tasks dynamically
_SUBJECT_MAPPINGS = {
    "accounting": "Accounting",
    "agriculture": "Agriculture",
    "architecture_and_engineering": "Architecture_and_Engineering",
    "art": "Art",
    "art_theory": "Art_Theory",
    "basic_medical_science": "Basic_Medical_Science",
    "biology": "Biology",
    "chemistry": "Chemistry",
    "clinical_medicine": "Clinical_Medicine",
    "computer_science": "Computer_Science",
    "design": "Design",
    "diagnostics_and_laboratory_medicine": "Diagnostics_and_Laboratory_Medicine",
    "economics": "Economics",
    "electronics": "Electronics",
    "energy_and_power": "Energy_and_Power",
    "finance": "Finance",
    "geography": "Geography",
    "history": "History",
    "literature": "Literature",
    "manage": "Manage",
    "marketing": "Marketing",
    "materials": "Materials",
    "math": "Math",
    "mechanical_engineering": "Mechanical_Engineering",
    "music": "Music",
    "pharmacy": "Pharmacy",
    "physics": "Physics",
    "psychology": "Psychology",
    "public_health": "Public_Health",
    "sociology": "Sociology",
}

# Create and export the subject-specific functions
for func_name, subject in _SUBJECT_MAPPINGS.items():
    globals()[f"mmmu_{func_name}"] = _create_subject_task(subject)

# Explicitly define exports for mypy
mmmu_accounting = globals()["mmmu_accounting"]
mmmu_agriculture = globals()["mmmu_agriculture"]
mmmu_architecture_and_engineering = globals()["mmmu_architecture_and_engineering"]
mmmu_art = globals()["mmmu_art"]
mmmu_art_theory = globals()["mmmu_art_theory"]
mmmu_basic_medical_science = globals()["mmmu_basic_medical_science"]
mmmu_biology = globals()["mmmu_biology"]
mmmu_chemistry = globals()["mmmu_chemistry"]
mmmu_clinical_medicine = globals()["mmmu_clinical_medicine"]
mmmu_computer_science = globals()["mmmu_computer_science"]
mmmu_design = globals()["mmmu_design"]
mmmu_diagnostics_and_laboratory_medicine = globals()[
    "mmmu_diagnostics_and_laboratory_medicine"
]
mmmu_economics = globals()["mmmu_economics"]
mmmu_electronics = globals()["mmmu_electronics"]
mmmu_energy_and_power = globals()["mmmu_energy_and_power"]
mmmu_finance = globals()["mmmu_finance"]
mmmu_geography = globals()["mmmu_geography"]
mmmu_history = globals()["mmmu_history"]
mmmu_literature = globals()["mmmu_literature"]
mmmu_manage = globals()["mmmu_manage"]
mmmu_marketing = globals()["mmmu_marketing"]
mmmu_materials = globals()["mmmu_materials"]
mmmu_math = globals()["mmmu_math"]
mmmu_mechanical_engineering = globals()["mmmu_mechanical_engineering"]
mmmu_music = globals()["mmmu_music"]
mmmu_pharmacy = globals()["mmmu_pharmacy"]
mmmu_physics = globals()["mmmu_physics"]
mmmu_psychology = globals()["mmmu_psychology"]
mmmu_public_health = globals()["mmmu_public_health"]
mmmu_sociology = globals()["mmmu_sociology"]


@task
def mmmu_mcq(
    subset: Optional[str] = None,
    split: str = "validation",
) -> Task:
    """MMMU Multiple-Choice only evaluation.

    Filters to multiple-choice questions and uses MCQ scoring.
    """
    dataset = get_dataset(subset=subset, split=split, question_type="multiple-choice")
    task_name = f"mmmu_mcq_{subset.lower()}" if subset else "mmmu_mcq"
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=create_mcq_scorer(group_keys=["subfield", "topic_difficulty"])(),
        name=task_name,
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )


@task
def mmmu_open(
    subset: Optional[str] = None,
    split: str = "validation",
) -> Task:
    """MMMU Open-answer only evaluation.

    Filters to open questions and scores via MMMU-style parser matching.
    """
    dataset = get_dataset(subset=subset, split=split, question_type="open")
    task_name = f"mmmu_open_{subset.lower()}" if subset else "mmmu_open"
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=mmmu_mixed_scorer(group_keys=["subfield", "topic_difficulty"])(),
        name=task_name,
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )
