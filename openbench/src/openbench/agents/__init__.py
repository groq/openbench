"""
Code agent implementations for openbench evaluations.

This module provides a unified interface for different CLI code agents
used in coding evaluations.
"""

from .base import BaseCodeAgent
from .aider import AiderAgent
from .opencode import OpenCodeAgent
from .claude import ClaudeAgent
from .roo import RooAgent
from .manager import AgentManager
from .docker_manager import DockerManager

__all__ = [
    "BaseCodeAgent",
    "AiderAgent",
    "OpenCodeAgent",
    "ClaudeAgent",
    "RooAgent",
    "AgentManager",
    "DockerManager",
]
