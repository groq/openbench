"""
SWE-bench evaluation for OpenBench.
Based on Princeton NLP's SWE-bench Verified dataset.
"""

import json
import logging
import platform
from pathlib import Path
from typing import List, Optional

from inspect_ai import task, Task
from inspect_ai.dataset import hf_dataset, FieldSpec
from inspect_ai.scorer import Scorer
from inspect_ai.util import SandboxEnvironmentSpec
from platformdirs import user_cache_dir

from .agent import swe_bench_agent, DEFAULT_BUNDLES
from .scorers import swe_bench_scorer

COMPOSE_FILES_DIR = Path(user_cache_dir("openbench_swebench")) / "compose_files"
logger = logging.getLogger(__name__)


@task
def swe_bench_verified() -> Task:
    """
    SWE-bench Verified evaluation task.

    Returns:
        Task: Configured SWE-bench Verified task for evaluating code generation on real-world issues.
    """
    return swe_bench(
        dataset="princeton-nlp/SWE-bench_Verified",
        split="test",
    )


@task
def swe_bench(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    instance_ids: Optional[List[str]] = None,
    scorer: Optional[Scorer | List[Scorer]] = None,
    token_limit: int = 1_000_000,
) -> Task:
    """
    Returns a Task for evaluating SWE-bench.

    Args:
        dataset: HuggingFace dataset name or local path.
        split: Dataset split to use.
        instance_ids: Optional list of instance IDs to filter.
        scorer: Optional custom scorer(s).
        token_limit: Maximum tokens for agent interaction.

    Returns:
        Task: Configured SWE-bench task.
    """
    # Load dataset samples
    samples = hf_dataset(
        dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "PASS_TO_PASS",
                "FAIL_TO_PASS",
                "test_patch",
                "version",
                "repo",
                "environment_setup_commit",
                "hints_text",
                "created_at",
            ],
        ),
        shuffle=True,
        seed=12345,
    )

    # Parse JSON fields
    for sample in samples:
        sample.metadata = sample.metadata or {}
        sample.metadata["PASS_TO_PASS"] = json.loads(sample.metadata["PASS_TO_PASS"])
        sample.metadata["FAIL_TO_PASS"] = json.loads(sample.metadata["FAIL_TO_PASS"])

    # Filter by instance IDs if specified
    if instance_ids is not None:
        samples = samples.filter(lambda x: x.id in instance_ids)

    # Get Docker image names for each instance
    ids_to_docker_image = get_image_names(samples)

    # Configure Docker environments
    for sample in samples:
        sample.metadata = sample.metadata or {}
        compose_file = get_compose_file(
            ids_to_docker_image[str(sample.id)], DEFAULT_BUNDLES
        )
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=compose_file,
        )
        sample.metadata["image"] = {
            "name": ids_to_docker_image[str(sample.id)],
        }

    return Task(
        dataset=samples,
        plan=swe_bench_agent(token_limit=token_limit, bundles=DEFAULT_BUNDLES),
        scorer=scorer or swe_bench_scorer(),
        epochs=1,
        metadata={"inspect-log-public": True},
    )


def get_compose_file(
    image_name: str, swe_agent_bundles: Optional[List[str]] = None
) -> str:
    """
    Create a Docker Compose file for the given image.

    Args:
        image_name: Docker image name to use.
        swe_agent_bundles: List of SWE-Agent bundles to include.

    Returns:
        str: Path to the compose file.
    """
    from .tools import generate_dockerfile_content

    # Create safe filenames
    base_filename = image_name.replace("/", "_").replace(":", "_")
    COMPOSE_FILES_DIR.mkdir(parents=True, exist_ok=True)

    # Create Dockerfile
    dockerfile_name = f"{base_filename}.Dockerfile"
    dockerfile_path = COMPOSE_FILES_DIR / dockerfile_name

    if swe_agent_bundles is None:
        dockerfile_content = f"FROM {image_name}"
    else:
        # Registry bundle is required by other bundles
        bundles_with_registry = ["registry"] + list(swe_agent_bundles)
        dockerfile_content = generate_dockerfile_content(
            image_name, bundles_with_registry
        )

    with dockerfile_path.open(mode="w+") as f:
        f.write(dockerfile_content)

    # Create compose file
    compose_filename = f"{base_filename}.yaml"
    image_compose_file = COMPOSE_FILES_DIR / compose_filename
    with image_compose_file.open(mode="w+") as f:
        f.write(f"""services:
  default:
    build:
      dockerfile: {dockerfile_name}
    command: "sleep infinity"
    working_dir: /testbed
    network_mode: none
    init: true""")

    return str(image_compose_file)


def get_image_names(samples) -> dict[str, str]:
    """Get Docker image names for each sample instance."""
    ids_to_docker_image = {}
    for instance in samples:
        if platform.machine() in ["x86_64", "AMD64"]:
            arch = "x86_64"
        elif platform.machine() in ["aarch64", "arm64"]:
            arch = "arm64"
            logger.warning(
                "Using arm64 architecture. Support is experimental. Some images may not exist or work."
            )
        else:
            arch = "x86_64"
            logger.warning(
                f"Unknown architecture {platform.machine()}, defaulting to x86_64"
            )

        image_name = f"ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance.id}"
        ids_to_docker_image[instance.id] = image_name

    return ids_to_docker_image
