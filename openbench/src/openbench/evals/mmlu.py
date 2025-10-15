from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE

# Adapted from https://github.com/openai/simple-evals

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


def record_to_mcq_sample(record: dict[str, str]) -> MCQSample:
    """Convert a MMLU record to an openbench MCQSample."""
    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["Question"],
            option_a=record["A"],
            option_b=record["B"],
            option_c=record["C"],
            option_d=record["D"],
        ),
        target=record["Answer"],
        metadata={
            "subject": record["Subject"],
            "category": SUBJECT_TO_CATEGORY[record["Subject"]],
        },
    )


@task
def mmlu(language: str = "EN_US") -> Task:
    """Evaluate the MMLU dataset. MCQ Abstracted."""
    if language == "EN_US":
        csv_file = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    else:
        raise ValueError(f"Language {language} not supported.")

    return MCQEval(
        name="mmlu",
        dataset_type="csv",
        dataset_path=csv_file,
        record_to_mcq_sample=record_to_mcq_sample,
        auto_id=True,
        config=GenerateConfig(
            temperature=0.5,
        ),
        group_keys=["category"],
    )
