"""
Create Inspect AI Tool wrappers for synthetic MCP tools.

This module provides wrappers that route tool calls to the synthetic HTTP MCP
server instead of real MCP servers. The HTTP server handles all tool execution
and returns deterministic responses.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.tool import Tool
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams
from inspect_ai.util._json import JSONSchema
import aiohttp


DEFAULT_HTTP_HOST = "localhost"
DEFAULT_HTTP_PORT = 8765


def _convert_json_schema(prop: dict[str, Any], name: str | None = None) -> JSONSchema:
    """Recursively convert a JSON schema dict to a JSONSchema object."""
    kwargs: dict[str, Any] = {}

    prop_type = prop.get("type")
    if isinstance(prop_type, list):
        kwargs["anyOf"] = [JSONSchema(type=t) for t in prop_type]  # type: ignore[arg-type]
    elif prop_type:
        kwargs["type"] = prop_type

    if "description" in prop:
        kwargs["description"] = prop["description"]
    elif name:
        kwargs["description"] = f"The {name} parameter"

    for field in ["format", "default", "enum"]:
        if field in prop:
            kwargs[field] = prop[field]

    if "items" in prop:
        items_schema = prop["items"]
        if isinstance(items_schema, dict):
            kwargs["items"] = _convert_json_schema(items_schema)

    if "properties" in prop:
        kwargs["properties"] = {
            k: _convert_json_schema(v, k) for k, v in prop["properties"].items()
        }

    if "required" in prop:
        kwargs["required"] = list(prop["required"])

    if "additionalProperties" in prop:
        ap = prop["additionalProperties"]
        if isinstance(ap, bool):
            kwargs["additionalProperties"] = ap
        elif isinstance(ap, dict):
            kwargs["additionalProperties"] = _convert_json_schema(ap)

    if "anyOf" in prop and "anyOf" not in kwargs:
        kwargs["anyOf"] = [_convert_json_schema(s) for s in prop["anyOf"]]

    return JSONSchema(**kwargs)  # type: ignore[arg-type]


def _json_schema_to_tool_params(schema: dict[str, Any]) -> ToolParams:
    """Convert a JSON schema to ToolParams."""
    properties_dict: dict[str, JSONSchema] = {}
    properties = schema.get("properties", {})
    required = list(schema.get("required", []))

    for name, prop in properties.items():
        properties_dict[name] = _convert_json_schema(prop, name)

    return ToolParams(
        type="object",
        properties=properties_dict,
        required=required,
    )


async def _call_http_tool(
    server_name: str,
    tool_name: str,
    params: dict[str, Any],
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call a tool on the synthetic HTTP MCP server.

    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        params: Tool parameters as a dictionary
        host: HTTP server host
        port: HTTP server port
        timeout: Request timeout in seconds

    Returns:
        Response dictionary with 'result' or 'error' key
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        try:
            url = f"http://{host}:{port}/mcp/{server_name}/tools/{tool_name}"
            async with session.post(
                url,
                json=params,
                headers={"Content-Type": "application/json"},
            ) as response:
                data = await response.json()
                return data
        except Exception as e:
            return {"error": str(e)}


def create_synthetic_tool_wrapper(
    server_name: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
    timeout: int = 30,
) -> Tool:
    """Create an Inspect AI Tool that routes to the synthetic HTTP MCP server.

    Args:
        server_name: Name of the MCP server (used in URL path)
        tool_name: Name of the tool
        tool_description: Description of the tool
        input_schema: JSON schema for tool parameters
        http_host: HTTP server host
        http_port: HTTP server port
        timeout: Timeout in seconds for tool execution

    Returns:
        Inspect AI Tool that executes via the HTTP MCP server
    """

    async def execute(**kwargs: Any) -> str:
        """Execute the tool via HTTP MCP server."""
        response = await _call_http_tool(
            server_name=server_name,
            tool_name=tool_name,
            params=kwargs,
            host=http_host,
            port=http_port,
            timeout=timeout,
        )

        if "error" in response and response["error"]:
            return f"Error: {response['error']}"

        result = response.get("result", "(no result)")

        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2)

    if input_schema:
        params = _json_schema_to_tool_params(input_schema)
    else:
        params = ToolParams(type="object", properties={}, required=[])

    safe_server_name = server_name.replace(" ", "_").replace("-", "_")
    safe_tool_name = tool_name.replace(" ", "_").replace("-", "_")
    prefixed_name = f"{safe_server_name}__{safe_tool_name}"

    tool_def = ToolDef(
        tool=execute,
        name=prefixed_name,
        description=tool_description,
        parameters=params,
        parallel=True,  # Synthetic tools are safe to run in parallel
    )

    return tool_def.as_tool()
