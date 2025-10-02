"""
OpenBench LiveMCPBench Copilot (ported)

This package provides a minimal port of the LiveMCPBench MCP Copilot agent
It exposes an MCP server that implements
two tools:

- "route": semantic tool/server selection based on a structured query
- "execute-tool": execution of a chosen tool on a chosen server

Credit: This implementation is adapted from the LiveMCPBench baseline
copilot agent:
https://github.com/icip-cas/LiveMCPBench/tree/main/baseline/mcp_copilot
"""

from .server import serve as run_copilot_server

__all__ = ["run_copilot_server"]
