"""OpenBench implementation of MuSR (Testing the Limits of Chain-of-thought with Multistep Soft Reasoning).
MuSR is a dataset that tests chain-of-thought reasoning with three types of tasks:
- Murder mysteries: Who is the most likely murderer?
- Object placements: Where would someone look for an object?
- Team allocation: How to allocate people to tasks efficiently?

Implemented by Aarush Sah
"""

import ast
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from openbench.scorers.musr import musr_grouped_scorer


def record_to_sample(record: dict, subset: str | None = None) -> Sample:
    # Parse the choices string representation into an actual list
    choices_list = ast.literal_eval(record["choices"])

    metadata = {
        "narrative": record["narrative"],
        "question": record["question"],
        "answer_choice": record["answer_choice"],
        "answer_index": record["answer_index"],
    }

    # Add subset metadata if provided
    if subset:
        metadata["subset"] = subset

    return Sample(
        input=f"{record['narrative']}\n\n{record['question']}",
        choices=choices_list,
        target=chr(ord("A") + int(record["answer_index"])),
        metadata=metadata,
    )


def create_combined_musr_dataset():
    """Create a combined dataset from all three MuSR subsets with subset metadata."""
    all_samples = []
    subsets = ["murder_mysteries", "object_placements", "team_allocation"]

    for subset in subsets:
        # Load each subset and add subset metadata
        subset_dataset = hf_dataset(
            path="TAUR-Lab/MuSR",
            split=subset,
            sample_fields=lambda record, s=subset: record_to_sample(record, s),
        )
        all_samples.extend(subset_dataset)

    return all_samples


@task
def musr(subset: str | None = None) -> Task:
    """
    MuSR (Multistep Soft Reasoning) evaluation task.

    Args:
        subset: The subset of the dataset to use. Options are:
                - None (default): Run all subsets with grouped metrics
                - "murder_mysteries": Murder mystery scenarios only
                - "object_placements": Object placement reasoning only
                - "team_allocation": Team allocation problems only
    """
    if subset is None:
        # Run all subsets with grouped metrics
        return Task(
            dataset=create_combined_musr_dataset(),
            solver=multiple_choice(),
            scorer=musr_grouped_scorer(),
        )
    else:
        # Run specific subset
        if subset not in ["murder_mysteries", "object_placements", "team_allocation"]:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: murder_mysteries, object_placements, team_allocation"
            )

        return Task(
            dataset=hf_dataset(
                path="TAUR-Lab/MuSR",
                split=subset,
                sample_fields=record_to_sample,
            ),
            solver=multiple_choice(),
            scorer=choice(),
        )


@task
def musr_murder_mysteries() -> Task:
    """MuSR Murder Mysteries - Who is the most likely murderer?"""
    return Task(
        dataset=hf_dataset(
            path="TAUR-Lab/MuSR",
            split="murder_mysteries",
            sample_fields=record_to_sample,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )


@task
def musr_object_placements() -> Task:
    """MuSR Object Placements - Where would someone look for an object?"""
    return Task(
        dataset=hf_dataset(
            path="TAUR-Lab/MuSR",
            split="object_placements",
            sample_fields=record_to_sample,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )


@task
def musr_team_allocation() -> Task:
    """MuSR Team Allocation - How to allocate people to tasks efficiently?"""
    return Task(
        dataset=hf_dataset(
            path="TAUR-Lab/MuSR",
            split="team_allocation",
            sample_fields=record_to_sample,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )
