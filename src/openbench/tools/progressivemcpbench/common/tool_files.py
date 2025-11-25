"""
Tool metadata loader for ProgressiveMCPBench.

Parses the upstream tools.json into a simpler structure that strategies can
use for listing and documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openbench.tools.progressivemcpbench.copilot.upstream_cache import (
    get_tools_json_cached,
)


@dataclass(frozen=True)
class ToolFile:
    """Represents a single tool surfaced as a virtual file."""

    server: str
    tool: str
    description: str
    input_schema: dict[str, Any]

    @property
    def filename(self) -> str:
        return f"{self.tool}.md"

    @property
    def path(self) -> str:
        return f"/{self.server}/{self.filename}"


def _iter_tools_raw(tools_json: List[Dict[str, Any]]) -> Iterable[Tuple[str, Dict]]:
    """Yield (server_name, server_info) pairs from tools.json content."""
    for tool_server in tools_json:
        tools_dict = tool_server.get("tools", {})
        if not isinstance(tools_dict, dict):
            continue
        for server_name, server_info in tools_dict.items():
            if not isinstance(server_info, dict):
                continue
            yield server_name, server_info


def load_tool_files(
    *, allow_servers: set[str] | None = None
) -> tuple[list[ToolFile], Path]:
    """Load tool metadata as ToolFile objects.

    Args:
        allow_servers: Optional whitelist of servers to include.

    Returns:
        (tool_files, source_path)
    """
    tools_list, cache_path = get_tools_json_cached()
    tool_files: list[ToolFile] = []
    allow = allow_servers or set()

    for server_name, server_info in _iter_tools_raw(tools_list):
        if allow and server_name not in allow:
            continue
        tools = server_info.get("tools", [])
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not name:
                continue
            desc = tool.get("description", "")
            schema = tool.get("inputSchema", {}) or {}
            try:
                tool_files.append(
                    ToolFile(
                        server=server_name,
                        tool=str(name),
                        description=str(desc),
                        input_schema=schema if isinstance(schema, dict) else {},
                    )
                )
            except Exception:
                # Skip malformed entries defensively
                continue

    return tool_files, cache_path
