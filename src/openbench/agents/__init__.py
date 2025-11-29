"""
Code agent implementations for openbench evaluations.

This module provides a unified interface for different CLI code agents
used in coding evaluations.
"""

from .base import BaseCodeAgent
from .aider import AiderAgent
from .opencode import OpenCodeAgent
from .gemini import GeminiAgent
from .roo import RooAgent
from .claude import ClaudeCodeAgent
from .codex import CodexAgent
from .manager import AgentManager
from .docker_manager import DockerManager

__all__ = [
    "BaseCodeAgent",
    "AiderAgent",
    "OpenCodeAgent",
    "GeminiAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "RooAgent",
    "AgentManager",
    "DockerManager",
]
