"""
MMLU-Redux - Manually re-annotated MMLU subset

MMLU-Redux is a subset of 5,700 manually re-annotated questions across 57 MMLU
subjects, addressing annotation errors in the original MMLU dataset.

Dataset: edinburgh-dawg/mmlu-redux-2.0
Paper: https://arxiv.org/abs/2406.04127

Sample usage:
```bash
bench eval mmlu_redux --model openrouter/openai/gpt-4o --limit 100
bench eval mmlu_redux_abstract_algebra --model groq/llama-3.1-70b
```

Citation:
@article{gema2024mmlu,
    title={Are We Done with MMLU?},
    author={Gema, Aryo Pradipta and Leang, Joshua Ong Jun and Hong, Giwon and
            Devoto, Alessio and Mber, Alberto Carlo Maria and Pinto, Francesco and others},
    journal={arXiv preprint arXiv:2406.04127},
    year={2024}
}
"""

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE

# Map MMLU subjects to categories (same as standard MMLU)
SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

MMLU_REDUX_SUBSETS = list(SUBJECT_TO_CATEGORY.keys())


def _make_record_to_mcq_sample(subject: str):
    """Create a record converter for a specific subject."""

    def record_to_mcq_sample(record: dict) -> MCQSample:
        """Convert a MMLU-Redux record to an MCQSample."""
        question = record["question"]
        choices = record["choices"]

        # Map answer index to letter
        answer_idx = (
            record["answer"]
            if isinstance(record["answer"], int)
            else int(record["answer"])
        )
        target = chr(65 + answer_idx)  # 0->A, 1->B, 2->C, 3->D

        # Build prompt with all choices
        prompt = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=question,
            option_a=choices[0],
            option_b=choices[1],
            option_c=choices[2],
            option_d=choices[3],
        )

        return MCQSample(
            input=prompt,
            target=target,
            metadata={
                "subject": subject,
                "category": SUBJECT_TO_CATEGORY[subject],
            },
        )

    return record_to_mcq_sample


@task
def mmlu_redux(subject: str | None = None) -> Task:
    """
    Evaluate MMLU-Redux benchmark - manually re-annotated MMLU subset.

    MMLU-Redux addresses annotation errors in the original MMLU dataset through
    manual re-annotation of 5,700 questions across all 57 subjects.

    Args:
        subject: Specific MMLU subject to evaluate. If None, evaluates all subjects.

    Returns:
        Task: Inspect AI task for MMLU-Redux evaluation
    """
    from inspect_ai.dataset import hf_dataset
    from inspect_ai.solver import generate
    from openbench.scorers.mcq import create_mcq_scorer

    if subject is not None:
        if subject not in SUBJECT_TO_CATEGORY:
            raise ValueError(
                f"Unknown subject '{subject}'. Valid subjects: {list[str](SUBJECT_TO_CATEGORY.keys())}"
            )
        subsets = [subject]
    else:
        subsets = MMLU_REDUX_SUBSETS

    # Load all requested subsets
    all_samples = []
    for subj in subsets:
        dataset = hf_dataset(
            "edinburgh-dawg/mmlu-redux-2.0",
            split="test",
            sample_fields=_make_record_to_mcq_sample(subj),
            auto_id=True,
            name=subj,
        )
        # Prefix IDs with subject to ensure uniqueness across combined dataset
        for sample in dataset:
            sample.id = f"{subj}_{sample.id}"
            all_samples.append(sample)

    from inspect_ai.dataset import MemoryDataset

    combined_dataset = MemoryDataset(samples=all_samples, name="mmlu_redux")

    return Task(
        name="mmlu_redux" if subject is None else f"mmlu_redux_{subject}",
        dataset=combined_dataset,
        solver=[generate()],
        scorer=create_mcq_scorer(group_keys=["category", "subject"])(),
        config=GenerateConfig(temperature=0.5),
    )


# Generate individual subject tasks for convenience
def _create_subject_task(subject: str):
    """Factory function to create a task for a specific subject."""

    @task
    def subject_task() -> Task:
        return mmlu_redux(subject=subject)

    subject_task.__name__ = f"mmlu_redux_{subject}"
    subject_task.__doc__ = (
        f"Evaluate MMLU-Redux on {subject.replace('_', ' ')} subject."
    )
    return subject_task


# Create individual task functions for each subject
mmlu_redux_abstract_algebra = _create_subject_task("abstract_algebra")
mmlu_redux_anatomy = _create_subject_task("anatomy")
mmlu_redux_astronomy = _create_subject_task("astronomy")
mmlu_redux_business_ethics = _create_subject_task("business_ethics")
mmlu_redux_clinical_knowledge = _create_subject_task("clinical_knowledge")
mmlu_redux_college_biology = _create_subject_task("college_biology")
mmlu_redux_college_chemistry = _create_subject_task("college_chemistry")
mmlu_redux_college_computer_science = _create_subject_task("college_computer_science")
mmlu_redux_college_mathematics = _create_subject_task("college_mathematics")
mmlu_redux_college_medicine = _create_subject_task("college_medicine")
mmlu_redux_college_physics = _create_subject_task("college_physics")
mmlu_redux_computer_security = _create_subject_task("computer_security")
mmlu_redux_conceptual_physics = _create_subject_task("conceptual_physics")
mmlu_redux_econometrics = _create_subject_task("econometrics")
mmlu_redux_electrical_engineering = _create_subject_task("electrical_engineering")
mmlu_redux_elementary_mathematics = _create_subject_task("elementary_mathematics")
mmlu_redux_formal_logic = _create_subject_task("formal_logic")
mmlu_redux_global_facts = _create_subject_task("global_facts")
mmlu_redux_high_school_biology = _create_subject_task("high_school_biology")
mmlu_redux_high_school_chemistry = _create_subject_task("high_school_chemistry")
mmlu_redux_high_school_computer_science = _create_subject_task(
    "high_school_computer_science"
)
mmlu_redux_high_school_european_history = _create_subject_task(
    "high_school_european_history"
)
mmlu_redux_high_school_geography = _create_subject_task("high_school_geography")
mmlu_redux_high_school_government_and_politics = _create_subject_task(
    "high_school_government_and_politics"
)
mmlu_redux_high_school_macroeconomics = _create_subject_task(
    "high_school_macroeconomics"
)
mmlu_redux_high_school_mathematics = _create_subject_task("high_school_mathematics")
mmlu_redux_high_school_microeconomics = _create_subject_task(
    "high_school_microeconomics"
)
mmlu_redux_high_school_physics = _create_subject_task("high_school_physics")
mmlu_redux_high_school_psychology = _create_subject_task("high_school_psychology")
mmlu_redux_high_school_statistics = _create_subject_task("high_school_statistics")
mmlu_redux_high_school_us_history = _create_subject_task("high_school_us_history")
mmlu_redux_high_school_world_history = _create_subject_task("high_school_world_history")
mmlu_redux_human_aging = _create_subject_task("human_aging")
mmlu_redux_human_sexuality = _create_subject_task("human_sexuality")
mmlu_redux_international_law = _create_subject_task("international_law")
mmlu_redux_jurisprudence = _create_subject_task("jurisprudence")
mmlu_redux_logical_fallacies = _create_subject_task("logical_fallacies")
mmlu_redux_machine_learning = _create_subject_task("machine_learning")
mmlu_redux_management = _create_subject_task("management")
mmlu_redux_marketing = _create_subject_task("marketing")
mmlu_redux_medical_genetics = _create_subject_task("medical_genetics")
mmlu_redux_miscellaneous = _create_subject_task("miscellaneous")
mmlu_redux_moral_disputes = _create_subject_task("moral_disputes")
mmlu_redux_moral_scenarios = _create_subject_task("moral_scenarios")
mmlu_redux_nutrition = _create_subject_task("nutrition")
mmlu_redux_philosophy = _create_subject_task("philosophy")
mmlu_redux_prehistory = _create_subject_task("prehistory")
mmlu_redux_professional_accounting = _create_subject_task("professional_accounting")
mmlu_redux_professional_law = _create_subject_task("professional_law")
mmlu_redux_professional_medicine = _create_subject_task("professional_medicine")
mmlu_redux_professional_psychology = _create_subject_task("professional_psychology")
mmlu_redux_public_relations = _create_subject_task("public_relations")
mmlu_redux_security_studies = _create_subject_task("security_studies")
mmlu_redux_sociology = _create_subject_task("sociology")
mmlu_redux_us_foreign_policy = _create_subject_task("us_foreign_policy")
mmlu_redux_virology = _create_subject_task("virology")
mmlu_redux_world_religions = _create_subject_task("world_religions")
