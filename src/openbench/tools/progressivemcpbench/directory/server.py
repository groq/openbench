"""
MCP server exposing ProgressiveMCPBench tools via a virtual directory listing.

Tools:
  - ls: list available servers or tool files
  - read-tool-file: read one or more tool markdown files
  - execute-tool: execute the underlying MCP tool referenced by a tool file path
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import Any, Iterable

import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP

from openbench.tools.progressivemcpbench.common.server_configs import (
    load_servers_from_config,
)
from openbench.tools.progressivemcpbench.common.tool_files import (
    ToolFile,
    load_tool_files,
)
from openbench.tools.progressivemcpbench.common.paths import rewrite_params_for_root
from openbench.tools.progressivemcpbench.copilot.mcp_connection import MCPConnection

logger = logging.getLogger(__name__)


def _normalize_path(path: str | None) -> str:
    """Normalize user-supplied paths to /server/tool.md form (allow /tools prefix)."""
    if not path:
        return "/"
    # Ensure leading slash and collapse repeats
    norm = str(PurePosixPath("/" + path.strip().lstrip("/")))
    if norm in {"/", ""}:
        return "/"
    # Strip optional /tools prefix to fit metaphor
    parts = norm.strip("/").split("/")
    if parts and parts[0].lower() == "tools":
        parts = parts[1:]
    rebuilt = "/" + "/".join(parts) if parts else "/"
    return rebuilt


class DirectoryIndex:
    """Virtual directory over available servers and tools."""

    def __init__(self, tool_files: list[ToolFile], servers: dict[str, Any]):
        # Only keep tools for servers we can execute
        allowed_servers = set(servers)
        self.tool_files = [
            tf for tf in tool_files if tf.server in allowed_servers
        ]
        self.servers = servers
        self.by_server: dict[str, list[ToolFile]] = {}
        self.path_map: dict[str, ToolFile] = {}

        for tf in self.tool_files:
            self.by_server.setdefault(tf.server, []).append(tf)
            self.path_map[tf.path] = tf
            # Allow paths without extension for convenience
            self.path_map[f"/{tf.server}/{tf.tool}"] = tf

        # Stable ordering for deterministic listings
        for values in self.by_server.values():
            values.sort(key=lambda t: t.tool)

        self.connection_lock = asyncio.Lock()

    def list_dir(self, path: str) -> list[str]:
        norm = _normalize_path(path)
        if norm in {"/", ""}:
            return sorted(f"{name}/" for name in self.by_server)

        # Paths are only one level deep (/server)
        parts = norm.strip("/").split("/")
        if len(parts) != 1:
            raise FileNotFoundError(f"Invalid directory: {path}")

        server = parts[0]
        if server not in self.by_server:
            raise FileNotFoundError(f"Server not found: {server}")

        return [tf.filename for tf in self.by_server[server]]

    def _resolve_tool(self, path: str) -> ToolFile:
        norm = _normalize_path(path)
        if norm.endswith("/"):
            norm = norm.rstrip("/")
        tf = self.path_map.get(norm)
        if not tf:
            # Also try with .md suffix
            if not norm.endswith(".md"):
                tf = self.path_map.get(norm + ".md")
        if not tf:
            raise FileNotFoundError(f"Tool file not found: {path}")
        return tf

    def read_files(self, paths: Iterable[str]) -> dict[str, ToolFile]:
        resolved: dict[str, ToolFile] = {}
        for p in paths:
            tf = self._resolve_tool(p)
            resolved[tf.path] = tf
        return resolved

    async def execute(self, path: str, params: dict[str, Any] | None) -> types.CallToolResult:
        tf = self._resolve_tool(path)
        server = self.servers.get(tf.server)
        if not server:
            raise ValueError(f"Server config missing for {tf.server}")

        async with self.connection_lock:
            async with MCPConnection(server) as connection:
                try:
                    rewritten = rewrite_params_for_root(params or {})
                    result = await asyncio.wait_for(
                        connection.call_tool(tf.tool, rewritten), timeout=300
                    )
                    return result
                except asyncio.TimeoutError:
                    return types.CallToolResult(
                        isError=True,
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"Tool {tf.tool} in {tf.server} call timed out.",
                            )
                        ],
                    )
                except Exception as e:  # pragma: no cover - runtime safety
                    error_msg = f"Error executing tool {tf.tool} on {tf.server}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return types.CallToolResult(
                        isError=True,
                        content=[types.TextContent(type="text", text=error_msg)],
                    )


def _format_schema(tf: ToolFile) -> str:
    props = tf.input_schema.get("properties", {}) if isinstance(tf.input_schema, dict) else {}
    required = set(tf.input_schema.get("required", []) if isinstance(tf.input_schema, dict) else [])
    lines: list[str] = []
    if props:
        lines.append("Parameters:")
        for name, spec in props.items():
            desc = spec.get("description", "") if isinstance(spec, dict) else ""
            dtype = spec.get("type", "") if isinstance(spec, dict) else ""
            default = spec.get("default")
            req = "required" if name in required else "optional"
            suffix = f" (default: {default})" if default is not None else ""
            lines.append(f"- {name} ({dtype}, {req}): {desc}{suffix}")
    else:
        lines.append("Parameters: none")
    return "\n".join(lines)


def _format_tool_markdown(tf: ToolFile) -> str:
    schema_json = (
        json.dumps(tf.input_schema, indent=2, ensure_ascii=False)
        if isinstance(tf.input_schema, dict)
        else "{}"
    )
    sections = [
        f"# {tf.tool} (server: {tf.server})",
        "",
        "Description:",
        tf.description.strip() or "(no description provided)",
        "",
        _format_schema(tf),
        "",
        "Input schema (JSON Schema):",
        schema_json,
    ]
    return "\n".join(sections)


_GLOBAL_INDEX: DirectoryIndex | None = None


def _get_index(ctx: "Context | None") -> DirectoryIndex:
    if ctx and ctx.request_context and ctx.request_context.lifespan_context:
        return ctx.request_context.lifespan_context["index"]  # type: ignore[return-value, index]
    if _GLOBAL_INDEX is None:
        raise RuntimeError("Directory index unavailable")
    return _GLOBAL_INDEX


def serve(config: dict[str, Any] | None = None) -> None:
    """Run the directory-style MCP server (stdio)."""
    servers = load_servers_from_config(config)
    tool_files, _ = load_tool_files(allow_servers=set(servers))
    index = DirectoryIndex(tool_files, servers)
    global _GLOBAL_INDEX
    _GLOBAL_INDEX = index

    if not index.tool_files:
        raise RuntimeError(
            "No tools available for the directory strategy. Ensure tools.json and clean_config.json are available."
        )

    @asynccontextmanager
    async def directory_lifespan(server: FastMCP):
        yield {"index": index}

    server = FastMCP("mcp-directory", lifespan=directory_lifespan)

    @server.tool(
        name="ls",
        description="List available servers (root) or tools within a server directory.",
    )
    async def ls(
        path: str | None = None,
        ctx: "Context | None" = None,
    ) -> types.CallToolResult:
        try:
            index_ctx = _get_index(ctx)
            entries = index_ctx.list_dir(path or "/")
            text = "\n".join(entries) if entries else "(empty)"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )
        except FileNotFoundError as e:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=str(e))]
            )

    @server.tool(
        name="read-tool-file",
        description="Read one or more tool markdown files to see descriptions and inputs.",
    )
    async def read_tool_file(
        paths: list[str] | str,
        ctx: "Context | None" = None,
    ) -> types.CallToolResult:
        try:
            index_ctx = _get_index(ctx)
            path_list = [paths] if isinstance(paths, str) else list(paths)
            resolved = index_ctx.read_files(path_list)
            chunks = []
            for path_key, tf in resolved.items():
                chunks.append(
                    f"{path_key}\n{'-' * len(path_key)}\n{_format_tool_markdown(tf)}"
                )
            text = "\n\n".join(chunks)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )
        except FileNotFoundError as e:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=str(e))]
            )

    @server.tool(
        name="execute-tool",
        description="Execute the tool referenced by the given tool file path.",
    )
    async def execute_tool(
        tool_path: str,
        params: dict[str, Any] | None = None,
        ctx: "Context | None" = None,
    ) -> types.CallToolResult:
        try:
            index_ctx = _get_index(ctx)
            return await index_ctx.execute(tool_path, params)
        except FileNotFoundError as e:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=str(e))]
            )
        except Exception as e:  # pragma: no cover - safety
            logger.error("execute-tool failed: %s", e, exc_info=True)
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=str(e))]
            )

    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
