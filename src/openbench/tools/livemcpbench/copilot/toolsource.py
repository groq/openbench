"""
ToolSource factory for the Copilot MCP server.

This creates an InspectAI ToolSource using mcp_server_stdio to run the
OpenBench-port Copilot server module and exposes its tools (route and execute-tool).
"""

from __future__ import annotations

import os
from typing import Optional

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools


def copilot_tool_source(
    python_executable: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> ToolSource:
    """Create a ToolSource for the Copilot MCP server.

    Args:
        python_executable: Optional path to Python to run the module
        extra_env: Additional environment variables to pass through

    Returns:
        ToolSource exposing route and execute-tool from Copilot
    """
    py: str = (
        python_executable
        if python_executable is not None
        else os.environ.get("PYTHON", "python")
    )
    passthrough_keys = {
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "OPENAI_API_KEY",
        "ABSTRACT_API_KEY",
        "ABSTRACT_BASE_URL",
        "ABSTRACT_MODEL",
        "MCP_DATA_PATH",
        "TOP_SERVERS",
        "TOP_TOOLS",
        "OPENBENCH_LIVEMCPBENCH_REFRESH",
        # Silence copilot logs
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
        "PLAYWRIGHT_BROWSERS_PATH",
    }
    env = {k: v for k, v in os.environ.items() if k in passthrough_keys}
    env.setdefault("OPENBENCH_COPILOT_SILENT", "1")
    # Common flags to suppress noisy child output
    env.setdefault("NODE_NO_WARNINGS", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("LOG_LEVEL", env.get("LOG_LEVEL", "error"))
    env.setdefault("RUST_LOG", env.get("RUST_LOG", "error"))
    env.setdefault("DEBUG", env.get("DEBUG", "0"))
    if extra_env:
        env.update(extra_env)

    server = mcp_server_stdio(
        command=py,
        args=["-m", "openbench.tools.livemcpbench.copilot.server"],
        env=env,
    )

    return mcp_tools(server, tools=["route", "execute-tool"])
