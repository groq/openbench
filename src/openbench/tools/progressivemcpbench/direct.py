"""
Direct tool exposure helpers for ProgressiveMCPBench strategies.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from inspect_ai.tool import ToolSource, mcp_server_stdio, mcp_tools

from openbench.tools.progressivemcpbench.common.server_configs import (
    load_servers_from_config,
)
from openbench.tools.progressivemcpbench.common.tool_files import (
    ToolFile,
    load_tool_files,
)

PROXY_ENV_LIST = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "PLAYWRIGHT_BROWSERS_PATH",
]

SILENT_DEFAULTS = {
    "NODE_NO_WARNINGS": "1",
    "PYTHONWARNINGS": "ignore",
    "NO_COLOR": "1",
    "LOG_LEVEL": "error",
    "RUST_LOG": "error",
    "DEBUG": "0",
    "PWDEBUG": "0",
}


def _build_env(base_env: dict[str, str] | None) -> dict[str, str]:
    env = dict(base_env or {})
    for proxy_env in PROXY_ENV_LIST:
        if proxy_env in os.environ:
            env[proxy_env] = os.environ[proxy_env]
    for k, v in SILENT_DEFAULTS.items():
        env.setdefault(k, v)
    return env


class DirectToolSourceFactory:
    """Build ToolSources for direct (non-meta) MCP server access."""

    def __init__(self):
        self.servers = load_servers_from_config()
        tool_files, _ = load_tool_files(allow_servers=set(self.servers))
        self.tool_files: list[ToolFile] = tool_files
        self.tools_by_server: dict[str, list[str]] = {}
        for tf in tool_files:
            self.tools_by_server.setdefault(tf.server, []).append(tf.tool)
        for tools in self.tools_by_server.values():
            tools.sort()

        self._toolsource_cache: dict[Tuple[str, Tuple[str, ...]], ToolSource] = {}

    def all_servers(self) -> set[str]:
        return set(self.servers)

    def server_tools(self, server: str) -> list[str]:
        return list(self.tools_by_server.get(server, []))

    def require_servers(self, servers: Iterable[str]) -> None:
        missing = [s for s in servers if s not in self.servers]
        if missing:
            raise ValueError(f"Unknown servers: {', '.join(missing)}")

    def require_tools(self, tools: Iterable[Tuple[str, str]]) -> None:
        unknown: list[str] = []
        for server, tool in tools:
            if server not in self.servers or tool not in self.tools_by_server.get(
                server, []
            ):
                unknown.append(f"{server}::{tool}")
        if unknown:
            raise ValueError(f"Unknown tools: {', '.join(unknown)}")

    def build_tool_source(
        self, server: str, tool_names: Sequence[str] | None = None
    ) -> ToolSource:
        """Return a ToolSource exposing a subset of tools for a server."""
        if server not in self.servers:
            raise ValueError(f"Server '{server}' is not available in config.")
        allowed = tuple(sorted(tool_names)) if tool_names else tuple()
        cache_key = (server, allowed)
        if cache_key in self._toolsource_cache:
            return self._toolsource_cache[cache_key]

        server_cfg = self.servers[server]
        env = _build_env(server_cfg.config.env)
        stdio = mcp_server_stdio(
            command=server_cfg.config.command,
            args=server_cfg.config.args,
            env=env,
        )
        tools = list(allowed) if allowed else "all"
        tool_source = mcp_tools(stdio, tools=tools)
        self._toolsource_cache[cache_key] = tool_source
        return tool_source

    def all_tool_tuples(self) -> list[Tuple[str, str]]:
        """Return all (server, tool) pairs in stable order."""
        tuples: list[Tuple[str, str]] = []
        for server in sorted(self.tools_by_server):
            for tool in self.tools_by_server[server]:
                tuples.append((server, tool))
        return tuples

    def deterministic_distractors(
        self,
        task_id: str,
        required: list[Tuple[str, str]],
        target_total: int = 128,
    ) -> list[Tuple[str, str]]:
        """Select deterministic distraction tools to reach target_total."""
        base_set = set(required)
        candidates = [
            pair for pair in self.all_tool_tuples() if pair not in base_set
        ]
        need = max(0, target_total - len(base_set))
        if need <= 0:
            return []

        scored = []
        for server, tool in candidates:
            digest = hashlib.sha256(f"{task_id}:{server}:{tool}".encode()).hexdigest()
            scored.append((digest, server, tool))
        scored.sort(key=lambda x: x[0])
        return [(server, tool) for _, server, tool in scored[:need]]


def parse_required_servers(metadata: dict[str, Any] | None) -> list[str]:
    raw = (metadata or {}).get("required_servers")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(s) for s in raw if s]
    return []


def parse_required_tools(
    metadata: dict[str, Any] | None,
) -> list[Tuple[str, str]]:
    raw = (metadata or {}).get("required_tools")
    if raw is None:
        return []
    tools: list[Tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                server = item.get("server_name") or item.get("server")
                tool = item.get("tool_name") or item.get("tool")
                if server and tool:
                    tools.append((str(server), str(tool)))
            elif isinstance(item, str):
                if "::" in item:
                    server, tool = item.split("::", 1)
                    tools.append((server, tool))
    elif isinstance(raw, dict):
        server = raw.get("server_name") or raw.get("server")
        tool = raw.get("tool_name") or raw.get("tool")
        if server and tool:
            tools.append((str(server), str(tool)))
    elif isinstance(raw, str):
        if "::" in raw:
            server, tool = raw.split("::", 1)
            tools.append((server, tool))
    return tools
