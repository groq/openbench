"""
ToolSource factory for the ProgressiveMCPBench Directory MCP server.

This creates an InspectAI ToolSource using mcp_server_stdio to run the
Synthetic Directory server module and exposes its tools (ls, read-tool-file, execute-tool).

The synthetic version routes tool execution to the HTTP MCP server instead of real MCP.
"""

from __future__ import annotations

import os
from typing import Optional

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools


def directory_tool_source(
    python_executable: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> ToolSource:
    """Create a ToolSource for the Synthetic Directory MCP server.

    This uses the synthetic directory server which routes tool execution
    to the HTTP MCP server instead of real MCP servers.

    Args:
        python_executable: Optional path to Python to run the module
        extra_env: Additional environment variables to pass through

    Returns:
        ToolSource exposing ls, read-tool-file, and execute-tool
    """
    py: str = (
        python_executable
        if python_executable is not None
        else os.environ.get("PYTHON", "python")
    )
    passthrough_keys = {
        "OPENBENCH_COPILOT_SILENT",
        "LOG_LEVEL",
        "RUST_LOG",
        "DEBUG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    env = {k: v for k, v in os.environ.items() if k in passthrough_keys}
    env.setdefault("OPENBENCH_COPILOT_SILENT", "1")
    env.setdefault("NODE_NO_WARNINGS", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("LOG_LEVEL", env.get("LOG_LEVEL", "error"))
    env.setdefault("RUST_LOG", env.get("RUST_LOG", "error"))
    env.setdefault("DEBUG", env.get("DEBUG", "0"))
    if extra_env:
        env.update(extra_env)

    # Use the synthetic directory server
    server = mcp_server_stdio(
        command=py,
        args=["-m", "openbench.tools.progressivemcpbench.synthetic.directory_server"],
        env=env,
    )

    return mcp_tools(
        server, tools=["meta__ls", "meta__read-tool-file", "meta__execute-tool"]
    )
