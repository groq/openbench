"""
ToolSource factory for the synthetic Copilot MCP server.

This creates an InspectAI ToolSource using mcp_server_stdio to run the
synthetic copilot server module and exposes its tools (route and execute-tool).
"""

from __future__ import annotations

import os
from typing import Optional

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools


def synthetic_copilot_tool_source(
    python_executable: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    http_host: str = "localhost",
    http_port: int = 9123,
) -> ToolSource:
    """Create a ToolSource for the synthetic Copilot MCP server.

    Args:
        python_executable: Optional path to Python to run the module
        extra_env: Additional environment variables to pass through
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource exposing route and execute-tool from synthetic Copilot
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
        "OPENBENCH_COPILOT_AUTOGEN",
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

    # Ensure autogen is enabled for synthetic embeddings
    env.setdefault("OPENBENCH_COPILOT_AUTOGEN", "1")
    env.setdefault("OPENBENCH_COPILOT_SILENT", "1")

    # Common flags to suppress noisy child output
    env.setdefault("NODE_NO_WARNINGS", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("LOG_LEVEL", env.get("LOG_LEVEL", "error"))
    env.setdefault("RUST_LOG", env.get("RUST_LOG", "error"))
    env.setdefault("DEBUG", env.get("DEBUG", "0"))

    # Pass HTTP server config
    env["SYNTHETIC_MCP_HOST"] = http_host
    env["SYNTHETIC_MCP_PORT"] = str(http_port)

    if extra_env:
        env.update(extra_env)

    server = mcp_server_stdio(
        command=py,
        args=["-m", "openbench.tools.progressivemcpbench.copilot.synthetic_server"],
        env=env,
    )

    return mcp_tools(server, tools=["meta__route", "meta__execute-tool"])
