"""CTI-Bench evaluation tasks for cybersecurity threat intelligence benchmarks."""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset, MemoryDataset

from openbench.datasets.cti_bench import (
    get_cti_bench_mcq_dataset,
    get_cti_bench_rcm_dataset,
    get_cti_bench_vsp_dataset,
    get_cti_bench_ate_dataset,
)
from openbench.scorers.cti_bench import (
    cti_bench_mcq_scorer,
    cti_bench_rcm_scorer,
    cti_bench_vsp_scorer,
    cti_bench_ate_scorer,
)


@task
def cti_bench_mcq() -> Task:
    """CTI-Bench Multiple Choice Questions task."""
    return Task(
        dataset=get_cti_bench_mcq_dataset(),
        solver=[generate()],
        scorer=cti_bench_mcq_scorer(),
        name="cti_bench_mcq",
        config=GenerateConfig(temperature=0.0, max_tokens=8192),
    )


@task
def cti_bench_rcm() -> Task:
    """CTI-Bench RCM (CVE→CWE mapping) task."""
    return Task(
        dataset=get_cti_bench_rcm_dataset(),
        solver=[generate()],
        scorer=cti_bench_rcm_scorer(),
        name="cti_bench_rcm",
        config=GenerateConfig(temperature=0.0, max_tokens=8192),
    )


@task
def cti_bench_vsp() -> Task:
    """CTI-Bench VSP (CVSS severity prediction) task."""
    return Task(
        dataset=get_cti_bench_vsp_dataset(),
        solver=[generate()],
        scorer=cti_bench_vsp_scorer(),
        name="cti_bench_vsp",
        config=GenerateConfig(temperature=0.0, max_tokens=8192),
    )


@task
def cti_bench_ate() -> Task:
    """CTI-Bench ATE (ATT&CK Technique Extraction) task."""
    return Task(
        dataset=get_cti_bench_ate_dataset(),
        solver=[generate()],
        scorer=cti_bench_ate_scorer(),
        name="cti_bench_ate",
        config=GenerateConfig(temperature=0.0, max_tokens=8192),
    )


def combine_datasets() -> Dataset:
    """Combine all CTI-Bench datasets into one."""
    # Load all individual datasets
    mcq_dataset = get_cti_bench_mcq_dataset()
    rcm_dataset = get_cti_bench_rcm_dataset()
    vsp_dataset = get_cti_bench_vsp_dataset()
    ate_dataset = get_cti_bench_ate_dataset()

    combined_samples = []

    # Add MCQ samples with task type metadata
    for sample in mcq_dataset:
        sample.metadata = sample.metadata or {}
        sample.metadata["task_type"] = "mcq"
        combined_samples.append(sample)

    # Add RCM samples with task type metadata
    for sample in rcm_dataset:
        sample.metadata = sample.metadata or {}
        sample.metadata["task_type"] = "rcm"
        combined_samples.append(sample)

    # Add VSP samples with task type metadata
    for sample in vsp_dataset:
        sample.metadata = sample.metadata or {}
        sample.metadata["task_type"] = "vsp"
        combined_samples.append(sample)

    # Add ATE samples with task type metadata
    for sample in ate_dataset:
        sample.metadata = sample.metadata or {}
        sample.metadata["task_type"] = "ate"
        combined_samples.append(sample)

    return MemoryDataset(samples=combined_samples, name="cti_bench")


def combined_cti_bench_scorer():
    """Combined scorer that delegates to appropriate task-specific scorer."""
    from inspect_ai.scorer import scorer, Score, Target, accuracy, stderr
    from inspect_ai.solver import TaskState
    from typing import Callable

    # Get individual scorers
    mcq_scorer_fn = cti_bench_mcq_scorer()
    rcm_scorer_fn = cti_bench_rcm_scorer()
    vsp_scorer_fn = cti_bench_vsp_scorer()
    ate_scorer_fn = cti_bench_ate_scorer()

    @scorer(metrics=[accuracy(), stderr()])
    def cti_bench_combined_scorer() -> Callable:
        async def score(state: TaskState, target: Target) -> Score:
            # Determine which scorer to use based on task type
            task_type = state.metadata.get("task_type") if state.metadata else None

            if task_type == "mcq":
                return await mcq_scorer_fn(state, target)
            elif task_type == "rcm":
                return await rcm_scorer_fn(state, target)
            elif task_type == "vsp":
                return await vsp_scorer_fn(state, target)
            elif task_type == "ate":
                return await ate_scorer_fn(state, target)
            else:
                # Fallback - should not happen
                return Score(
                    value=0.0,
                    answer="",
                    metadata={"error": f"Unknown task type: {task_type}"},
                )

        return score

    return cti_bench_combined_scorer()


@task
def cti_bench() -> Task:
    """Combined CTI-Bench evaluation running all 4 cybersecurity tasks."""
    return Task(
        dataset=combine_datasets(),
        solver=[generate()],
        scorer=combined_cti_bench_scorer(),
        name="cti_bench",
        config=GenerateConfig(temperature=0.0, max_tokens=8192),
    )
