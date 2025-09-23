"""
Docker container management for code agents.

This module provides dynamic Docker container generation based on agent requirements,
enabling efficient, agent-specific containers instead of monolithic ones.
"""

from __future__ import annotations

import hashlib
import subprocess
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .manager import AgentManager
from openbench.provider_config import ProviderManager
from openbench.utils.docker import (
    BASE_DOCKERFILE_TEMPLATE,
    COMMON_BASE_PACKAGES,
    LANGUAGE_RUNTIME_PACKAGES,
    EVAL_COMPOSE_TEMPLATE,
    STANDARD_COMPOSE_TEMPLATE,
)


class DockerManager:
    """Manager for dynamic Docker container generation based on code agents."""

    @classmethod
    def generate_dockerfile(cls, agent_name: str) -> str:
        """Generate a Dockerfile for the specific agent.

        Args:
            agent_name: Name of the code agent

        Returns:
            Complete Dockerfile content as string

        Raises:
            ValueError: If agent name is not supported
        """
        agent = AgentManager.get_agent(agent_name)

        # Get agent requirements
        base_packages = cls._get_all_base_packages(agent)
        agent_commands = cls._format_agent_commands(agent)
        runtime_setup = cls._format_runtime_setup(agent)

        # Generate Dockerfile
        dockerfile = BASE_DOCKERFILE_TEMPLATE.format(
            base_packages=" ".join(base_packages),
            agent_commands=agent_commands,
            runtime_setup=runtime_setup,
        )

        return dockerfile

    @classmethod
    def get_container_tag(cls, agent_name: str) -> str:
        """Get the Docker tag for an agent's container.

        Args:
            agent_name: Name of the code agent

        Returns:
            Docker tag string
        """
        # Create a hash of the agent's requirements for versioning
        agent = AgentManager.get_agent(agent_name)
        requirements_hash = cls._hash_agent_requirements(agent)

        return f"openbench-{agent_name}:{requirements_hash[:8]}"

    @classmethod
    def generate_eval_files(cls, agent_name: str, output_dir: Path) -> None:
        """Generate Dockerfile and compose.yaml for inspect_ai evaluation.

        This method creates agent-specific Docker files that inspect_ai can use
        for its sandbox environment. The files are written to the output directory
        and will be picked up by inspect_ai when the task is executed.

        Uses smart caching to avoid regenerating identical files.

        Args:
            agent_name: Name of the code agent
            output_dir: Directory to write the files (e.g., /evals/exercism/)
        """
        dockerfile_path = output_dir / "Dockerfile"
        compose_path = output_dir / "compose.yaml"

        # Generate new content
        dockerfile_content = cls.generate_dockerfile(agent_name)
        compose_content = cls._generate_eval_compose_content(agent_name)

        # Check if files need updating
        dockerfile_needs_update = True
        compose_needs_update = True

        if dockerfile_path.exists():
            existing_dockerfile = dockerfile_path.read_text()
            dockerfile_needs_update = existing_dockerfile != dockerfile_content

        if compose_path.exists():
            existing_compose = compose_path.read_text()
            compose_needs_update = existing_compose != compose_content

        # Only write if content has changed
        if dockerfile_needs_update:
            dockerfile_path.write_text(dockerfile_content)

        if compose_needs_update:
            compose_path.write_text(compose_content)

    @classmethod
    def generate_compose_file(
        cls, agent_name: str, output_path: Optional[Path] = None
    ) -> Path:
        """Generate Docker Compose file for the specific agent.

        Args:
            agent_name: Name of the code agent
            output_path: Optional path for compose file (defaults to temp file)

        Returns:
            Path to the generated compose file
        """
        # Get container tag for the agent
        container_tag = cls.get_container_tag(agent_name)
        agent = AgentManager.get_agent(agent_name)

        # Generate compose content
        compose_content = cls._generate_compose_content(
            agent_name, container_tag, agent
        )

        # Write to file
        if output_path is None:
            # Create temp file that won't be deleted immediately
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f".{agent_name}.yaml",
                prefix="docker-compose.",
                delete=False,
            )
            output_path = Path(temp_file.name)
            temp_file.write(compose_content)
            temp_file.close()
        else:
            output_path.write_text(compose_content)

        print(f"📝 Generated compose file: {output_path}")
        return output_path

    @classmethod
    def _get_all_base_packages(cls, agent) -> List[str]:
        """Get all base packages needed including common ones."""
        agent_packages = agent.get_base_packages()

        # Combine and deduplicate
        all_packages = list(
            dict.fromkeys(
                COMMON_BASE_PACKAGES + LANGUAGE_RUNTIME_PACKAGES + agent_packages
            )
        )
        return all_packages

    @classmethod
    def _format_agent_commands(cls, agent) -> str:
        """Format agent installation commands for Dockerfile."""
        commands = agent.get_dockerfile_commands()
        if not commands:
            return "# No agent-specific installation required"

        return "\n".join(commands)

    @classmethod
    def _format_runtime_setup(cls, agent) -> str:
        """Format runtime setup commands for Dockerfile."""
        runtime_commands = agent.get_runtime_commands()
        if not runtime_commands:
            return "# No runtime setup required"

        formatted_commands = []
        for cmd in runtime_commands:
            formatted_commands.append(f"RUN {cmd}")

        return "\n".join(formatted_commands)

    @classmethod
    def _hash_agent_requirements(cls, agent) -> str:
        """Create hash of agent requirements for versioning."""
        # Combine all agent requirements into a single string
        requirements = {
            "dockerfile_commands": agent.get_dockerfile_commands(),
            "base_packages": agent.get_base_packages(),
            "volumes": agent.get_volumes(),
            "runtime_commands": agent.get_runtime_commands(),
            "env_requirements": agent.get_env_requirements(),
        }

        requirements_str = str(sorted(requirements.items()))
        return hashlib.sha256(requirements_str.encode()).hexdigest()

    @classmethod
    def _container_exists(cls, tag: str) -> bool:
        """Check if Docker container with given tag exists."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", tag], capture_output=True, text=True
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @classmethod
    def _generate_eval_compose_content(cls, agent_name: str) -> str:
        """Generate Docker Compose file content for inspect_ai evaluation.

        This version uses 'build: .' with a cache tag for better Docker caching.
        The tag helps Docker reuse containers when the agent requirements haven't changed.

        Args:
            agent_name: Name of the code agent

        Returns:
            Complete compose.yaml content as string
        """
        agent = AgentManager.get_agent(agent_name)

        # Get deterministic tag for caching
        container_tag = cls.get_container_tag(agent_name)

        # Format environment variables (provider + agent-specific)
        env_vars = cls._format_environment_variables(agent)
        environment_section = f"    environment:\n{env_vars}\n" if env_vars else ""

        # Format volumes
        volumes_section = cls._format_volumes_section(agent)
        volume_definitions = cls._format_volume_definitions(agent)

        compose_content = EVAL_COMPOSE_TEMPLATE.format(
            container_tag=container_tag,
            environment_section=environment_section,
            volumes_section=volumes_section,
            volume_definitions=volume_definitions,
        )
        return compose_content

    @classmethod
    def _generate_compose_content(
        cls, agent_name: str, container_tag: str, agent
    ) -> str:
        """Generate Docker Compose file content."""
        # Format environment variables
        env_vars = cls._format_environment_variables(agent)
        environment_section = f"    environment:\n{env_vars}\n" if env_vars else ""

        # Format volumes
        volumes_section = cls._format_volumes_section(agent)
        volume_definitions = cls._format_volume_definitions(agent)

        compose_content = STANDARD_COMPOSE_TEMPLATE.format(
            container_tag=container_tag,
            environment_section=environment_section,
            volumes_section=volumes_section,
            volume_definitions=volume_definitions,
        )
        return compose_content

    @classmethod
    def _format_environment_variables(cls, agent) -> str:
        """Format environment variables for compose file."""
        # Get all provider environment variables
        provider_env_vars = ProviderManager.get_all_env_vars()

        # Include only provider envs that have non-empty values on host
        host_env_vars = [
            env for env in provider_env_vars if (os.getenv(env) not in (None, ""))
        ]

        # Get agent-specific environment variables
        agent_env_vars = agent.get_env_requirements()

        # Combine and deduplicate
        all_env_vars = list(dict.fromkeys(host_env_vars + agent_env_vars))

        if not all_env_vars:
            return ""

        env_lines = [f"      {var}: ${{{var}:-}}" for var in all_env_vars]
        return "\n".join(env_lines)

    @classmethod
    def _format_volumes_section(cls, agent) -> str:
        """Format volumes section for compose file."""
        volumes = agent.get_volumes()
        if not volumes:
            return ""

        volume_lines = ["    volumes:"]
        for volume_name, mount_path in volumes.items():
            volume_lines.append(f"      - {volume_name}:{mount_path}")

        return "\n".join(volume_lines)

    @classmethod
    def _format_volume_definitions(cls, agent) -> str:
        """Format volume definitions for compose file."""
        volumes = agent.get_volumes()
        if not volumes:
            return ""

        volume_lines = ["volumes:"]
        for volume_name in volumes.keys():
            volume_lines.append(f"  {volume_name}:")

        return "\n".join(volume_lines)
