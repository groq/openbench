"""CTI-Bench evaluation tasks for cybersecurity threat intelligence benchmarks."""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig

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
