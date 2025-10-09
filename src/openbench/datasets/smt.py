from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from typing import Optional


SMT_PROMPT = """
Solve the following Stanford Math Tournament problem step by step. Show your work clearly and provide your final answer.

{question}

Remember to put your final answer on its own line after "Answer:", without using \\boxed command.
""".strip()


def record_to_sample(record: dict) -> Sample:
    """Convert an SMT record to an Inspect Sample."""
    task = SMT_PROMPT.format(question=record["question"])
    answer = record.get("answer", "")
    # Ensure answer is not None
    if answer is None:
        answer = ""
    category = record.get("category", "")
    problem_id = record.get("id", "")

    return Sample(
        input=task,
        target=str(answer),
        metadata={
            "category": category,
            "id": problem_id,
        },
    )


def get_smt_dataset(category: Optional[str] = None) -> Dataset:
    """Load the SMT 2024 dataset.

    Args:
        category: Optional category filter (Algebra, Calculus, Discrete, General, Geometry, Guts)

    Returns:
        Dataset containing SMT problems
    """
    dataset = hf_dataset(
        path="nmayorga7/smt-2024",
        split="train",
        sample_fields=record_to_sample,
    )
    samples = list(dataset)
    if category is not None:
        samples = [
            sample
            for sample in samples
            if (sample.metadata or {}).get("category") == category
        ]
    name = f"smt_{category.lower()}" if category is not None else "smt"
    return MemoryDataset(samples=samples, name=name)


def get_smt_algebra_dataset() -> Dataset:
    """Load the SMT dataset filtered for Algebra problems."""
    return get_smt_dataset(category="Algebra")


def get_smt_calculus_dataset() -> Dataset:
    """Load the SMT dataset filtered for Calculus problems."""
    return get_smt_dataset(category="Calculus")


def get_smt_discrete_dataset() -> Dataset:
    """Load the SMT dataset filtered for Discrete problems."""
    return get_smt_dataset(category="Discrete")


def get_smt_general_dataset() -> Dataset:
    """Load the SMT dataset filtered for General problems."""
    return get_smt_dataset(category="General")


def get_smt_geometry_dataset() -> Dataset:
    """Load the SMT dataset filtered for Geometry problems."""
    return get_smt_dataset(category="Geometry")


def get_smt_guts_dataset() -> Dataset:
    """Load the SMT dataset filtered for Guts round problems."""
    return get_smt_dataset(category="Guts")
