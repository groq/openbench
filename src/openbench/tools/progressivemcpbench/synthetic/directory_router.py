"""
Router for directory-based tool discovery and execution using synthetic MCP.

This router presents tools as a filesystem structure where servers are directories
and tools are markdown files within those directories. Tool execution goes through
the HTTP MCP server.
"""

from __future__ import annotations

import json
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mcp.types as types

from ..mcp_config import get_mcp_base_url


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _load_servers_config() -> dict[str, Any]:
    """Load the synthetic servers.json configuration."""
    config_path = _synthetic_mcp_dir() / "config" / "servers.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


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


class SyntheticDirectoryRouter:
    """Router that presents synthetic tools as a directory structure."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
    ):
        """Initialize the synthetic directory router.

        Args:
            base_url: Base URL for the MCP server (defaults to configured URL)
            timeout: Timeout for HTTP requests
        """
        self.base_url = base_url if base_url is not None else get_mcp_base_url()
        self.timeout = timeout

        # Load tools from servers.json
        servers_config = _load_servers_config()
        self.tools_index: dict[str, dict[str, dict[str, Any]]] = {}
        self.server_descriptions: dict[str, str] = {}

        for server_name, server in servers_config.items():
            if server_name not in self.tools_index:
                self.tools_index[server_name] = {}
            self.server_descriptions[server_name] = server.get("description", "")
            for tool in server.get("tools", []):
                tool_name = tool.get("name", "")
                if tool_name:
                    self.tools_index[server_name][tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {}),
                    }

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

    def _call_http_tool(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a tool on the synthetic HTTP MCP server using MCP protocol."""
        parsed = urlparse(self.base_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if use_https else 80)

        conn: HTTPConnection | HTTPSConnection
        if use_https:
            conn = HTTPSConnection(host, port, timeout=self.timeout)
        else:
            conn = HTTPConnection(host, port, timeout=self.timeout)

        call_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
        }
        body = json.dumps(call_data)

        try:
            conn.request(
                "POST",
                f"/mcp/{server_name}",
                body,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response = conn.getresponse()
            response_text = response.read().decode("utf-8")

            try:
                if "data: " in response_text:
                    lines = response_text.strip().split("\n")
                    data_line = None
                    for line in lines:
                        if line.startswith("data: "):
                            data_line = line[6:]
                            break

                    if data_line:
                        data = json.loads(data_line)
                    else:
                        return {"error": "No data field in SSE response"}
                else:
                    data = json.loads(response_text)

                if "error" in data:
                    return {"error": data["error"].get("message", str(data["error"]))}

                return data

            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse response: {str(e)}"}

        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    async def execute_tool(
        self,
        tool_path: str,
        params: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        """Execute a tool by its path via HTTP MCP server.

        Args:
            tool_path: Path to the tool file (e.g., "/tools/filesystem/read_file.md")
            params: Parameters to pass to the tool

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

        response = self._call_http_tool(server_name, tool_name, params or {})

        if "error" in response and response["error"]:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text=f"Error: {response['error']}")
                ],
            )

        result = response.get("result", "(no result)")
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, indent=2)

        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> "SyntheticDirectoryRouter":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
