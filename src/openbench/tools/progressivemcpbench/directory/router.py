"""
Router for directory-based tool discovery and execution.

This router presents MCP tools as a filesystem structure where servers are directories
and tools are markdown files within those directories.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import mcp.types as types

from ..copilot.schemas import Server, ServerConfig
from ..copilot.mcp_connection import MCPConnection

logger = logging.getLogger(__name__)


def _root_sandbox_dir() -> Path:
    return Path(os.path.expanduser("~/.openbench/progressivemcpbench/root")).resolve()


def _rewrite_root_path(value: str) -> str:
    if value.startswith("/root/"):
        base = _root_sandbox_dir()
        rel = value[len("/root/") :]
        return str(base / rel)
    return value


def _rewrite_params_for_root(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite_params_for_root(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_params_for_root(v) for v in obj]
    if isinstance(obj, str):
        new_path = _rewrite_root_path(obj)
        if new_path != obj:
            p = Path(new_path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return new_path
    return obj


def _truncate_description(description: str, max_len: int = 120) -> str:
    """Truncate description to first line or max_len chars."""
    if not description:
        return ""
    first_line = description.split("\n")[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[:max_len]


def _format_input_schema(schema: dict[str, Any]) -> str:
    """Format a JSON schema as a readable markdown description."""
    if not schema:
        return "No parameters required."

    lines = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for name, prop in properties.items():
        prop_type = prop.get("type", "any")
        description = prop.get("description", "")
        req_marker = " (required)" if name in required else " (optional)"
        lines.append(f"- **{name}**: `{prop_type}`{req_marker}")
        if description:
            lines.append(f"  {description}")

    return "\n".join(lines) if lines else "No parameters required."


class DirectoryRouter:
    """Router that presents tools as a directory structure."""

    def __init__(self, config: dict[str, Any], tools_data: list[dict[str, Any]]):
        """Initialize the directory router.

        Args:
            config: MCP server configuration (mcpServers dict)
            tools_data: List of server tool specifications from tools.json
        """
        self.servers: dict[str, Server] = {}
        self.tools_index: dict[str, dict[str, dict[str, Any]]] = {}
        self.server_descriptions: dict[str, str] = {}

        sandbox = _root_sandbox_dir()
        mcp_servers = config.get("mcpServers", {})

        for name, config_data in mcp_servers.items():
            cfg = dict(config_data)
            args = list(cfg.get("args", []))
            if name.lower() in {
                "filesystem",
                "server-filesystem",
                "@modelcontextprotocol/server-filesystem",
            }:
                if args and args[-1] == "/":
                    args[-1] = str(sandbox)
            for i, a in enumerate(args):
                if isinstance(a, str) and "annotated_data" in a:
                    parts = a.split("annotated_data/", 1)
                    if len(parts) == 2:
                        args[i] = str(sandbox / parts[1])
            cfg["args"] = args
            self.servers[name] = Server(name=name, config=ServerConfig(**cfg))

        # tools_data format: list of server entries with nested tools structure
        # Each entry has: name, description, tools: {server_name: {tools: [...]}}
        for entry in tools_data:
            tools_dict = entry.get("tools", {})
            # tools_dict is keyed by server name
            for server_name, server_info in tools_dict.items():
                if not server_name:
                    continue
                if server_name not in self.tools_index:
                    self.tools_index[server_name] = {}
                # Store server description from the entry level
                if server_name not in self.server_descriptions:
                    self.server_descriptions[server_name] = entry.get("description", "")
                for tool in server_info.get("tools", []):
                    tool_name = tool.get("name", "")
                    if tool_name:
                        self.tools_index[server_name][tool_name] = {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "inputSchema": tool.get("inputSchema", {}),
                        }

        self.connection_lock = asyncio.Lock()

    def list_path(self, path: str) -> str:
        """List contents of a path in the virtual directory structure.

        Args:
            path: Path to list (e.g., "/tools" or "/tools/filesystem")

        Returns:
            String listing of directory contents
        """
        path = path.rstrip("/")

        if path == "" or path == "/":
            return "tools/"

        if path == "/tools":
            servers = sorted(self.tools_index.keys())
            if not servers:
                return "(empty directory)"
            lines = []
            for s in servers:
                desc = _truncate_description(self.server_descriptions.get(s, ""))
                if desc:
                    lines.append(f"{s}/ # {desc}")
                else:
                    lines.append(f"{s}/")
            return "\n".join(lines)

        if path.startswith("/tools/"):
            parts = path[7:].split("/")
            server_name = parts[0]

            if server_name not in self.tools_index:
                raise ValueError(f"Server not found: {server_name}")

            tools = self.tools_index[server_name]
            if not tools:
                return "(empty directory)"
            lines = []
            for name in sorted(tools.keys()):
                desc = _truncate_description(tools[name].get("description", ""))
                if desc:
                    lines.append(f"{name}.md # {desc}")
                else:
                    lines.append(f"{name}.md")
            return "\n".join(lines)

        raise ValueError(f"Invalid path: {path}")

    def read_tool_file(self, path: str) -> str:
        """Read a tool file and return its markdown description.

        Args:
            path: Path to the tool file (e.g., "/tools/filesystem/read_file.md")

        Returns:
            Markdown description of the tool including its input schema
        """
        if not path.startswith("/tools/"):
            raise ValueError(f"Invalid tool path: {path}")

        parts = path[7:].split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid tool path format: {path}")

        server_name = parts[0]
        tool_file = parts[1]

        if not tool_file.endswith(".md"):
            raise ValueError(f"Tool file must end with .md: {tool_file}")

        tool_name = tool_file[:-3]

        if server_name not in self.tools_index:
            raise ValueError(f"Server not found: {server_name}")

        if tool_name not in self.tools_index[server_name]:
            raise ValueError(f"Tool not found: {tool_name} in {server_name}")

        tool = self.tools_index[server_name][tool_name]

        lines = [
            f"# {tool_name}",
            "",
            f"**Server:** {server_name}",
            "",
            "## Description",
            "",
            tool.get("description", "No description available."),
            "",
            "## Parameters",
            "",
            _format_input_schema(tool.get("inputSchema", {})),
        ]

        return "\n".join(lines)

    def _parse_tool_path(self, tool_path: str) -> tuple[str, str]:
        """Parse a tool path into server name and tool name.

        Args:
            tool_path: Path like "/tools/filesystem/read_file.md"

        Returns:
            Tuple of (server_name, tool_name)
        """
        if not tool_path.startswith("/tools/"):
            raise ValueError(f"Invalid tool path: {tool_path}")

        parts = tool_path[7:].split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid tool path format: {tool_path}")

        server_name = parts[0]
        tool_file = parts[1]

        if tool_file.endswith(".md"):
            tool_name = tool_file[:-3]
        else:
            tool_name = tool_file

        return server_name, tool_name

    async def execute_tool(
        self,
        tool_path: str,
        params: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> types.CallToolResult:
        """Execute a tool by its path.

        Args:
            tool_path: Path to the tool file (e.g., "/tools/filesystem/read_file.md")
            params: Parameters to pass to the tool
            timeout: Timeout in seconds

        Returns:
            Tool execution result
        """
        server_name, tool_name = self._parse_tool_path(tool_path)

        if server_name not in self.tools_index:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(
                        type="text", text=f"Server not found: {server_name}"
                    )
                ],
            )

        if tool_name not in self.tools_index[server_name]:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text=f"Tool not found: {tool_name}")
                ],
            )

        async with self.connection_lock:
            server_config = self.servers.get(server_name)
            if not server_config:
                return types.CallToolResult(
                    isError=True,
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Server '{server_name}' is not defined in the configuration.",
                        )
                    ],
                )
            async with MCPConnection(server_config) as connection:
                try:
                    rewritten = _rewrite_params_for_root(params or {})
                    result = await asyncio.wait_for(
                        connection.call_tool(tool_name, rewritten), timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    return types.CallToolResult(
                        isError=True,
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"Tool {tool_name} in {server_name} call timed out.",
                            )
                        ],
                    )
                except Exception as e:
                    error_msg = (
                        f"Error executing tool {tool_name} on {server_name}: {str(e)}"
                    )
                    return types.CallToolResult(
                        isError=True,
                        content=[types.TextContent(type="text", text=error_msg)],
                    )

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
