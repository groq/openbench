"""
MCP Directory server entrypoint using synthetic MCP backend.

This strategy presents tools as a directory structure where:
- /tools/ contains server directories
- /tools/<server>/ contains tool files (markdown descriptions)
- The model can use ls, read-tool-file, and execute-tool to explore and use tools

Tool execution is routed to the HTTP MCP server instead of real MCP servers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP

from .directory_router import SyntheticDirectoryRouter
from ..mcp_config import get_mcp_base_url

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Reduce noisy logs by default."""
    silent = os.getenv("OPENBENCH_COPILOT_SILENT", "1") in {"1", "true", "True"}
    if not silent:
        return
    logging.basicConfig(level=logging.WARNING)
    for name in [
        "mcp",
        "mcp.client",
        "mcp.server",
        "httpx",
        "urllib3",
        "anyio",
        "asyncio",
        "fastmcp",
        __name__,
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


_GLOBAL_ROUTER: SyntheticDirectoryRouter | None = None


def serve(
    base_url: str | None = None,
) -> None:
    """Run the Synthetic Directory MCP server (stdio).

    Args:
        base_url: Base URL for the MCP server (defaults to configured URL)
    """
    _configure_logging()
    resolved_base_url = base_url if base_url is not None else get_mcp_base_url()

    @asynccontextmanager
    async def directory_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        router = SyntheticDirectoryRouter(base_url=resolved_base_url)
        async with router:
            global _GLOBAL_ROUTER
            _GLOBAL_ROUTER = router
            try:
                yield {"router": router}
            finally:
                _GLOBAL_ROUTER = None

    server = FastMCP("mcp-synthetic-directory", lifespan=directory_lifespan)

    @server.tool(
        name="meta__ls",
        description=(
            """
List the contents of a path in the tool directory.

Use this to explore available MCP servers and their tools.
- ls("/tools") - lists all available server directories
- ls("/tools/<server>") - lists all tool files for a specific server

Each server is a directory containing markdown files that describe individual tools.

IMPORTANT: You MUST call meta__read-tool-file to read a tool's full description and parameters BEFORE calling meta__execute-tool for that tool. Never execute a tool without first reading its specification.
"""
        ),
    )
    async def ls(
        path: str,
        ctx: Context | None = None,
    ) -> types.CallToolResult:
        router = (
            ctx.request_context.lifespan_context["router"]  # type: ignore[union-attr]
            if ctx is not None
            else _GLOBAL_ROUTER
        )
        if router is None:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text="Router context unavailable")
                ],
            )
        try:
            result = router.list_path(path)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result)]
            )
        except Exception as e:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
            )

    @server.tool(
        name="meta__read-tool-file",
        description=(
            """
Read the description and parameters of one or more tools.

Parameters:
    paths: A single path string or a list of paths to tool files.
           Each path should be in the format: /tools/<server>/<tool>.md

Returns the tool description and input schema in markdown format.
Multiple paths can be provided to read several tools in one call for efficiency.

Example:
    read-tool-file("/tools/filesystem/read_file.md")
    read-tool-file(["/tools/filesystem/read_file.md", "/tools/filesystem/write_file.md"])
"""
        ),
    )
    async def read_tool_file(
        paths: str | list[str],
        ctx: Context | None = None,
    ) -> types.CallToolResult:
        router = (
            ctx.request_context.lifespan_context["router"]  # type: ignore[union-attr]
            if ctx is not None
            else _GLOBAL_ROUTER
        )
        if router is None:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text="Router context unavailable")
                ],
            )
        try:
            if isinstance(paths, str):
                paths = [paths]
            results = []
            for path in paths:
                content = router.read_tool_file(path)
                results.append(f"## {path}\n\n{content}")
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text="\n\n---\n\n".join(results))
                ]
            )
        except Exception as e:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
            )

    @server.tool(
        name="meta__execute-tool",
        description=(
            """
Execute a tool from the directory.

Parameters:
    tool_path: The path to the tool file (e.g., /tools/<server>/<tool>.md)
    params: A dictionary of parameters to pass to the tool

Example:
    execute-tool(tool_path="/tools/filesystem/read_file.md", params={"path": "/root/test.txt"})
"""
        ),
    )
    async def execute_tool(
        tool_path: str,
        params: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> types.CallToolResult:
        router = (
            ctx.request_context.lifespan_context["router"]  # type: ignore[union-attr]
            if ctx is not None
            else _GLOBAL_ROUTER
        )
        if router is None:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text="Router context unavailable")
                ],
            )
        result = await router.execute_tool(tool_path, params)
        return result

    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
