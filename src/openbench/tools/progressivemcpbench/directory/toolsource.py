"""
ToolSource factory for the directory-style ProgressiveMCPBench MCP server.
"""

from __future__ import annotations

import os
from typing import Optional

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools


def directory_tool_source(
    python_executable: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> ToolSource:
    """Create a ToolSource for the directory-based MCP server."""
    py: str = (
        python_executable
        if python_executable is not None
        else os.environ.get("PYTHON", "python")
    )
    passthrough_keys = {
        "OPENAI_API_KEY",
        "ABSTRACT_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "MCP_DATA_PATH",
        "TOP_SERVERS",
        "TOP_TOOLS",
        "OPENBENCH_PROGRESSIVEMCPBENCH_REFRESH",
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
    env.setdefault("NODE_NO_WARNINGS", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("LOG_LEVEL", env.get("LOG_LEVEL", "error"))
    env.setdefault("RUST_LOG", env.get("RUST_LOG", "error"))
    env.setdefault("DEBUG", env.get("DEBUG", "0"))
    env.setdefault("NO_COLOR", "1")
    if extra_env:
        env.update(extra_env)

    server = mcp_server_stdio(
        command=py,
        args=["-m", "openbench.tools.progressivemcpbench.directory.server"],
        env=env,
    )

    return mcp_tools(server, tools=["ls", "read-tool-file", "execute-tool"])
