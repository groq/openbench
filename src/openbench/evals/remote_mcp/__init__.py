"""
Remote MCP handlers for provider-specific server-side MCP implementations.

This module provides a unified interface for executing tasks using remote MCP
servers across different providers (Groq, Anthropic, etc.).
"""

from openbench.evals.remote_mcp.base import RemoteMCPHandler
from openbench.evals.remote_mcp.registry import get_remote_mcp_handler

__all__ = ["RemoteMCPHandler", "get_remote_mcp_handler"]
