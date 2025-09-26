"""OpenBench implementation of MuSR (Testing the Limits of Chain-of-thought with Multistep Soft Reasoning).
MuSR is a dataset that tests chain-of-thought reasoning with three types of tasks:
- Murder mysteries: Who is the most likely murderer?
- Object placements: Where would someone look for an object?
- Team allocation: How to allocate people to tasks efficiently?

Implemented by Aarush Sah
"""

import ast
from typing import Optional
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, MemoryDataset
from inspect_ai.solver import generate
from openbench.utils.mcq import MCQSample, MCQEval
from openbench.utils.text import create_dynamic_multiple_choice_prompt
from openbench.scorers.mcq import create_mcq_scorer


def record_to_mcq_sample(record: dict, subset: Optional[str] = None) -> MCQSample:
    """Convert a MuSR record to an OpenBench MCQSample."""
    try:
        choices_list = ast.literal_eval(record["choices"])  # type: ignore[arg-type]
    except Exception:
        choices_list = []

    question_text = f"{record['narrative']}\n\n{record['question']}"
    prompt = create_dynamic_multiple_choice_prompt(question_text, choices_list)

    metadata = {
        "narrative": record.get("narrative", ""),
        "question": record.get("question", ""),
        "answer_choice": record.get("answer_choice", ""),
        "answer_index": record.get("answer_index", ""),
    }
    if subset:
        metadata["subset"] = subset

    try:
        target_letter = chr(ord("A") + int(record["answer_index"]))
    except Exception:
        target_letter = "A"

    return MCQSample(
        input=prompt,
        target=target_letter,
        metadata=metadata,
    )


def create_combined_musr_dataset():
    """Create a combined dataset from all three MuSR subsets with subset metadata."""
    all_samples = []
    subsets = ["murder_mysteries", "object_placements", "team_allocation"]

    for subset in subsets:
        subset_dataset = hf_dataset(
            path="TAUR-Lab/MuSR",
            split=subset,
            sample_fields=lambda record, s=subset: record_to_mcq_sample(record, s),
        )
        all_samples.extend(subset_dataset)

    return MemoryDataset(samples=all_samples, name="musr_combined")


@task
def musr(subset: Optional[str] = None) -> Task:
    """
    MuSR (Multistep Soft Reasoning) evaluation task.

    Args:
        subset: The subset of the dataset to use. Options are:
                - None (default): Run all subsets with grouped metrics
                - "murder_mysteries": Murder mystery scenarios only
                - "object_placements": Object placement reasoning only
                - "team_allocation": Team allocation problems only
    """
    valid_subsets = ["murder_mysteries", "object_placements", "team_allocation"]
    if subset is None:
        return Task(
            dataset=create_combined_musr_dataset(),
            solver=[generate()],
            scorer=create_mcq_scorer(group_keys=["subset"])(),
            name="musr",
        )
    else:
        if subset not in valid_subsets:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: {', '.join(valid_subsets)}"
            )

        return MCQEval(
            name=f"musr_{subset}",
            dataset_path="TAUR-Lab/MuSR",
            record_to_mcq_sample=record_to_mcq_sample,
            split=subset,
        )


@task
def musr_murder_mysteries() -> Task:
    """MuSR Murder Mysteries - Who is the most likely murderer?"""
    return musr(subset="murder_mysteries")


@task
def musr_object_placements() -> Task:
    """MuSR Object Placements - Where would someone look for an object?"""
    return musr(subset="object_placements")


@task
def musr_team_allocation() -> Task:
    """MuSR Team Allocation - How to allocate people to tasks efficiently?"""
    return musr(subset="team_allocation")
