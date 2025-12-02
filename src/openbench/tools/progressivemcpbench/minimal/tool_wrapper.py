"""
Create Inspect AI Tool wrappers for MCP tools.

This module provides a way to wrap individual MCP tools as Inspect AI Tool
objects, allowing them to be used directly by the agent without the
route/execute-tool indirection.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from inspect_ai.tool import Tool
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams
from inspect_ai.util._json import JSONSchema
import mcp.types as types

from ..copilot.schemas import Server
from ..copilot.mcp_connection import MCPConnection


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


def _format_result(result: types.CallToolResult) -> str:
    """Format a CallToolResult as a string."""
    if result.isError:
        error_text = ""
        for content in result.content:
            if isinstance(content, types.TextContent):
                error_text += content.text
        return f"Error: {error_text}" if error_text else "Error: Unknown error"

    text_parts = []
    for content in result.content:
        if isinstance(content, types.TextContent):
            text_parts.append(content.text)
        elif isinstance(content, types.ImageContent):
            text_parts.append(f"[Image: {content.mimeType}]")
        elif isinstance(content, types.EmbeddedResource):
            text_parts.append(f"[Resource: {content.resource}]")

    return "\n".join(text_parts) if text_parts else "(no output)"


def _convert_json_schema(prop: dict[str, Any], name: str | None = None) -> JSONSchema:
    """Recursively convert a JSON schema dict to a JSONSchema object.

    Handles nested schemas (items, properties) and union types (anyOf).
    """
    kwargs: dict[str, Any] = {}

    # Handle union types like ["integer", "null"]
    prop_type = prop.get("type")
    if isinstance(prop_type, list):
        # Convert to anyOf format
        kwargs["anyOf"] = [
            JSONSchema(type=t)  # type: ignore[arg-type]
            for t in prop_type
        ]
    elif prop_type:
        kwargs["type"] = prop_type

    # Description (provide default if at top level with a name)
    if "description" in prop:
        kwargs["description"] = prop["description"]
    elif name:
        kwargs["description"] = f"The {name} parameter"

    # Simple scalar fields
    for field in ["format", "default", "enum"]:
        if field in prop:
            kwargs[field] = prop[field]

    # Nested items schema (for arrays)
    if "items" in prop:
        items_schema = prop["items"]
        if isinstance(items_schema, dict):
            kwargs["items"] = _convert_json_schema(items_schema)

    # Nested properties (for objects)
    if "properties" in prop:
        kwargs["properties"] = {
            k: _convert_json_schema(v, k) for k, v in prop["properties"].items()
        }

    # Required fields for object types
    if "required" in prop:
        kwargs["required"] = list(prop["required"])

    # Additional properties
    if "additionalProperties" in prop:
        ap = prop["additionalProperties"]
        if isinstance(ap, bool):
            kwargs["additionalProperties"] = ap
        elif isinstance(ap, dict):
            kwargs["additionalProperties"] = _convert_json_schema(ap)

    # anyOf (if already present in source)
    if "anyOf" in prop and "anyOf" not in kwargs:
        kwargs["anyOf"] = [_convert_json_schema(s) for s in prop["anyOf"]]

    return JSONSchema(**kwargs)  # type: ignore[arg-type]


def _json_schema_to_tool_params(schema: dict[str, Any]) -> ToolParams:
    """Convert a JSON schema to ToolParams.

    Recursively converts nested schemas to ensure compatibility with
    strict API validators like OpenAI.
    """
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


_connection_lock = asyncio.Lock()


def create_mcp_tool_wrapper(
    server: Server,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    timeout: int = 300,
) -> Tool:
    """Create an Inspect AI Tool that wraps an MCP tool.

    Args:
        server: Server configuration
        tool_name: Name of the tool (will be prefixed with server name)
        tool_description: Description of the tool
        input_schema: JSON schema for tool parameters
        timeout: Timeout in seconds for tool execution

    Returns:
        Inspect AI Tool that executes the MCP tool
    """
    server_name = server.name

    async def execute(**kwargs: Any) -> str:
        """Execute the MCP tool with the provided arguments."""
        async with _connection_lock:
            async with MCPConnection(server) as connection:
                try:
                    rewritten = _rewrite_params_for_root(kwargs)
                    result = await asyncio.wait_for(
                        connection.call_tool(tool_name, rewritten), timeout=timeout
                    )
                    return _format_result(result)
                except asyncio.TimeoutError:
                    return f"Error: Tool {tool_name} timed out after {timeout}s"
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"

    # Create ToolParams from the JSON schema
    if input_schema:
        params = _json_schema_to_tool_params(input_schema)
    else:
        params = ToolParams(type="object", properties={}, required=[])

    # Prefix tool name with server name to avoid collisions
    # e.g., "filesystem__read_file" instead of just "read_file"
    # Replace whitespace with underscores to avoid API errors
    safe_server_name = server_name.replace(" ", "_")
    safe_tool_name = tool_name.replace(" ", "_")
    prefixed_name = f"{safe_server_name}__{safe_tool_name}"

    # Create ToolDef and convert to Tool
    tool_def = ToolDef(
        tool=execute,
        name=prefixed_name,
        description=tool_description,
        parameters=params,
        parallel=False,  # MCP tools use a shared connection lock
    )

    return tool_def.as_tool()
