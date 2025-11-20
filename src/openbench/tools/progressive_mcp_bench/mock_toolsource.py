from __future__ import annotations
import os
import sys
from typing import Optional

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools

def mock_mcp_tool_source(
    db_path: str,
    strategy: str = "all",
    relevant_server: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> ToolSource:
    py = python_executable or sys.executable
    
    # Pass environment variables to the subprocess
    env = os.environ.copy()
    env.update({
        "PROGRESSIVE_MCP_DB": db_path,
        "PROGRESSIVE_MCP_STRATEGY": strategy,
        "OPENBENCH_COPILOT_SILENT": "1",
        "NODE_NO_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore",
        "LOG_LEVEL": "error",
        "RUST_LOG": "error",
        "DEBUG": "0",
    })
    
    if relevant_server:
        env["PROGRESSIVE_MCP_RELEVANT_SERVER"] = relevant_server

    server = mcp_server_stdio(
        command=py,
        args=["-m", "openbench.tools.progressive_mcp_bench.mock_server"],
        env=env,
    )
    
    tools_arg = ["route", "execute-tool"]
    if strategy == "all":
        tools_arg = "all"
        
    return mcp_tools(server, tools=tools_arg)
