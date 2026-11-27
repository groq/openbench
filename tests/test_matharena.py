"""Tests for MathArena task scorer selection."""

import importlib

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai._util.registry import registry_info

from openbench.evals.matharena import matharena


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """Skip the Hugging Face download when building tasks."""
    monkeypatch.setattr(
        matharena,
        "hf_dataset",
        lambda **kwargs: MemoryDataset([Sample(input="1+1?", target="2")]),
    )


def scorer_name(task) -> str:
    return registry_info(task.scorer[0]).name.split("/")[-1]


def build(task_name: str):
    module = importlib.import_module(
        f"openbench.evals.matharena.{task_name}.{task_name}"
    )
    return getattr(module, task_name)()


def test_default_scorer_is_aime():
    task = matharena.matharena_task(
        dataset_path="x/y", instruction="Solve.", name="t", default_max_tokens=10
    )
    assert scorer_name(task) == "aime_scorer"


@pytest.mark.parametrize(
    "task_name", ["aime_2025", "aime_2025_II", "aime_2024_I", "aime_2023_I"]
)
def test_aime_tasks_keep_aime_scorer(task_name):
    assert scorer_name(build(task_name)) == "aime_scorer"


@pytest.mark.parametrize(
    "task_name", ["brumo_2025", "hmmt_feb_2023", "hmmt_feb_2024", "hmmt_feb_2025"]
)
def test_non_aime_tasks_use_boxed_expression_scorer(task_name):
    assert scorer_name(build(task_name)) == "boxed_expression_scorer"
