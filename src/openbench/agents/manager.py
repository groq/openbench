"""
Agent registry for Inspect SWE-powered code agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

from inspect_ai.agent import Agent
from inspect_swe import claude_code, codex_cli


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for a supported code agent."""

    name: str
    builder: Callable[..., Agent]
    default_model: str
    description: str
    aliases: Sequence[str] = ()
    required_env: Sequence[str] = ()
    default_attempts: int = 1


class AgentManager:
    """Lightweight registry for Inspect SWE agents."""

    _agents: Dict[str, AgentSpec] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def _register(cls, spec: AgentSpec) -> None:
        """Register an agent specification."""
        cls._agents[spec.name] = spec
        for alias in spec.aliases:
            cls._aliases[alias] = spec.name

    @classmethod
    def _normalize(cls, agent_name: str) -> str:
        """Normalize user input to a canonical agent name."""
        key = agent_name.lower()
        if key in cls._agents:
            return key
        if key in cls._aliases:
            return cls._aliases[key]
        raise ValueError(f"Unsupported code agent: {agent_name}")

    @classmethod
    def get_agent_spec(cls, agent_name: str) -> AgentSpec:
        """Return the agent specification for the given name."""
        return cls._agents[cls._normalize(agent_name)]

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        *,
        cwd: str,
        model: str | None = None,
        attempts: int | None = None,
        system_prompt: str | None = None,
    ) -> Agent:
        """Instantiate an Inspect SWE agent with the provided options."""
        spec = cls.get_agent_spec(agent_name)
        kwargs: Dict[str, Any] = {"cwd": cwd}
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if model:
            kwargs["model"] = model
        if attempts or spec.default_attempts != 1:
            kwargs["attempts"] = attempts or spec.default_attempts
        return spec.builder(**kwargs)

    @classmethod
    def get_supported_agents(cls) -> List[str]:
        """List all supported agent names."""
        # ensure canonical names sorted for stable help text
        return sorted(cls._agents.keys())

    @classmethod
    def get_valid_code_agents(cls) -> List[str]:
        """Alias for get_supported_agents (backwards compatibility)."""
        return cls.get_supported_agents()

    @classmethod
    def is_valid_agent(cls, agent_name: str) -> bool:
        """Return True if the agent name or alias is recognized."""
        try:
            cls._normalize(agent_name)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_code_agent(cls, agent_name: str) -> bool:
        """Backwards-compatible validation wrapper."""
        return cls.is_valid_agent(agent_name)

    @classmethod
    def get_default_model(cls, agent_name: str) -> str:
        """Default benchmark model for an agent."""
        return cls.get_agent_spec(agent_name).default_model

    @classmethod
    def get_description(cls, agent_name: str) -> str:
        """Human-readable description."""
        return cls.get_agent_spec(agent_name).description

    @classmethod
    def resolve_model(cls, agent_name: str, requested_model: str | None = None) -> str:
        """Return the requested model or fall back to the default."""
        if requested_model:
            return requested_model
        return cls.get_default_model(agent_name)

    @classmethod
    def get_help_text(cls) -> str:
        """Help text for `--code-agent` CLI flag."""
        agent_names = ", ".join(cls.get_supported_agents())
        default_agent = "codex"
        return (
            f"Inspect SWE code agent to run inside the Exercism sandbox. "
            f"Options: {agent_names} (default: {default_agent})"
        )


# Register supported agents
AgentManager._register(
    AgentSpec(
        name="codex",
        builder=codex_cli,
        default_model=os.getenv("OPENBENCH_CODEX_MODEL", "openai/gpt-5"),
        description="Inspect SWE Codex CLI agent",
        aliases=("codex_cli",),
    )
)

AgentManager._register(
    AgentSpec(
        name="claude_code",
        builder=claude_code,
        default_model=os.getenv(
            "OPENBENCH_CLAUDE_CODE_MODEL", "anthropic/claude-sonnet-4-5-20250929"
        ),
        description="Inspect SWE Claude Code agent",
        aliases=("claude",),
        required_env=("ANTHROPIC_API_KEY",),
    )
)
