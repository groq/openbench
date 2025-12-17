"""
Streamable HTTP MCP Client

Implements the MCP Streamable HTTP Transport protocol as described in the MCP spec:
- Uses HTTP POST with JSON responses
- Handles MCP initialization, tools listing, and tool execution
- Designed for the synthetic MCP server on localhost:9123
"""

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import mcp.types as types
from http.client import HTTPConnection

logger = logging.getLogger(__name__)


class StreamableHTTPClient:
    """MCP client using Streamable HTTP Transport protocol."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._connected: bool = False
        self._tools_cache: Dict[str, Any] = {}

    def _get_connection(self) -> HTTPConnection:
        """Create HTTP connection to the MCP server."""
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return HTTPConnection(host, port, timeout=self.timeout)

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the MCP server."""
        conn = self._get_connection()
        try:
            url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
            body = json.dumps(data) if data else "{}"
            headers = {"Content-Type": "application/json"}

            conn.request(method, url, body, headers)
            response = conn.getresponse()
            response_data = response.read().decode("utf-8")

            try:
                return json.loads(response_data)
            except json.JSONDecodeError:
                return {
                    "error": f"Invalid JSON response: {response_data}",
                    "status": response.status,
                }
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
        finally:
            conn.close()

    def initialize(self) -> bool:
        """Initialize the MCP connection."""
        init_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "openbench-synthetic-client",
                    "version": "1.0.0",
                },
            },
        }

        response = self._make_request("POST", "/initialize", init_data)
        if "error" in response:
            logger.error(f"MCP initialization error: {response['error']}")
            return False

        self._connected = True
        logger.info("MCP connection initialized successfully")
        return True

    def list_tools(self) -> List[types.Tool]:
        """List available tools from the MCP server."""
        if not self._connected:
            if not self.initialize():
                return []

        # Remove the leading server path and get tools list
        server_path = (
            self.base_url.split("/mcp/")[-1] if "/mcp/" in self.base_url else ""
        )
        if not server_path:
            return []

        tools_data = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = self._make_request("POST", f"/mcp/{server_path}", tools_data)

        if "error" in response:
            logger.error(f"Tools list error: {response['error']}")
            return []

        result = response.get("result", [])
        tools = []

        for tool_data in result:
            # Convert MCP tool format to types.Tool
            tool = types.Tool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {}),
            )
            tools.append(tool)
            # Cache the tools
            self._tools_cache[tool.name] = tool_data

        logger.info(f"Listed {len(tools)} tools from server: {server_path}")
        return tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with arguments."""
        if not self._connected:
            if not self.initialize():
                return {"error": "Failed to initialize MCP connection"}

        # Extract server name from URL
        server_path = (
            self.base_url.split("/mcp/")[-1] if "/mcp/" in self.base_url else ""
        )
        if not server_path:
            return {"error": "No server specified in base URL"}

        call_data = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = self._make_request("POST", f"/mcp/{server_path}", call_data)

        if "error" in response:
            logger.error(f"Tool call error: {response['error']}")
            return {"error": response["error"]}

        result = response.get("result", {})
        logger.info(f"Called tool {tool_name} on server {server_path}")
        return result

    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected
