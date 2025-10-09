# Ported from https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/cybench
# See LICENSE

import logging
from pathlib import Path
from typing import Callable, Literal

from inspect_ai import Task, task
from inspect_ai.agent import AgentAttempts, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.tool import bash, python
from inspect_ai.util import SandboxEnvironmentSpec
from openbench.utils.text import (
    CYBENCH_CONTINUE_MESSAGE,
    CYBENCH_DOCKER_WARNING,
    CYBENCH_INCORRECT_MESSAGE,
    CYBENCH_SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)

CHALLENGES_DIR = Path(__file__).parent / "challenges"


@task
def cybench(
    eval_names: str | list[str] | None = None,
    variant_names: str | list[str] | None = "hard",
    sandbox_type: Literal["docker", "k8s"] = "docker",
) -> Task:
    """
    Create a task for the Cybench challenges.

    Args:
        eval_names (str | list[str] | None): The list of evals to load.
        variant_names (str | list[str] | None): The variant or list of variants to
            include. If None, all variants are included.
        sandbox_type (str): The type of sandbox to use.
    """
    from inspect_cyber import create_agentic_eval_dataset  # type: ignore[import-not-found]

    if sandbox_type == "docker":
        _validate_docker_sandbox_usage(logger)

    dataset = (
        create_agentic_eval_dataset(
            root_dir=CHALLENGES_DIR.absolute(),
        )
        .filter_by_metadata_field("eval_name", eval_names)
        .filter_by_metadata_field("variant_name", variant_names)
    )
    if sandbox_type == "k8s":
        import importlib.util

        if importlib.util.find_spec("k8s_sandbox") is None:
            logger.warning(
                "Using the k8s sandbox with Cybench requires the inspect_k8s_sandbox package"
                " to convert the compose.yaml files, but it is not installed.\n\n"
                "Please install it with: pip install inspect_k8s_sandbox\n\n"
                "See: https://k8s-sandbox.aisi.org.uk/helm/compose-to-helm/ for more information."
            )

        dataset = dataset.flat_map(_convert_sandbox("k8s"))

    return Task(
        dataset=dataset,
        solver=react(
            prompt=CYBENCH_SYSTEM_MESSAGE,
            tools=[bash(timeout=180), python(timeout=180)],
            attempts=AgentAttempts(
                attempts=3,
                incorrect_message=CYBENCH_INCORRECT_MESSAGE,
            ),
            on_continue=CYBENCH_CONTINUE_MESSAGE,
        ),
        scorer=includes(),
    )


def _validate_docker_sandbox_usage(logger: logging.Logger) -> None:
    """Raise an error if Docker is used without acknowledgment."""
    import os

    if not os.environ.get("CYBENCH_ACKNOWLEDGE_RISKS"):
        raise ValueError(
            f"{CYBENCH_DOCKER_WARNING}\n\n"
            "Set CYBENCH_ACKNOWLEDGE_RISKS=1 in your environment to proceed."
        )

    logger.warning(f"\n{CYBENCH_DOCKER_WARNING}")


def _convert_sandbox(sandbox_type: str) -> Callable[[Sample], list[Sample]]:
    """Create a mapping function to convert a sample's sandbox to the given type."""

    def mapper(sample: Sample) -> list[Sample]:
        if sample.sandbox is None:
            new_sandbox = SandboxEnvironmentSpec(type=sandbox_type)
        elif isinstance(sample.sandbox, SandboxEnvironmentSpec):
            new_sandbox = SandboxEnvironmentSpec(
                type=sandbox_type, config=sample.sandbox.config
            )
        else:
            raise ValueError(f"Unexpected sandbox type: {sample.sandbox}")

        new_sample = sample.model_copy(
            deep=True,
            update={"sandbox": new_sandbox},
        )
        return [new_sample]

    return mapper
