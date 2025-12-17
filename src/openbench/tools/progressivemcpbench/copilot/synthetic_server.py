"""
Synthetic MCP Copilot server for ProgressiveMCPBench.

This server exposes the same route/execute-tool interface as the live copilot,
but uses the synthetic HTTP MCP backend for tool execution.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
import asyncio
import json
import logging
import os

import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP

from .synthetic_router import SyntheticCopilotRouter
from .synthetic_arg_generation import (
    generate_synthetic_embeddings,
    get_synthetic_embeddings_path,
    synthetic_embeddings_exist,
)
from ..mcp_config import get_mcp_base_url

logger = logging.getLogger(__name__)


def _configure_logging():
    """Reduce noisy logs from Copilot and underlying libraries by default."""
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


# Global router fallback (used if ctx isn't provided by runtime)
_GLOBAL_ROUTER: "SyntheticCopilotRouter | None" = None


def serve(
    base_url: str | None = None,
) -> None:
    """Run the Synthetic Copilot MCP server (stdio).

    Args:
        base_url: Base URL for the MCP server (defaults to configured URL)
    """
    _configure_logging()
    resolved_base_url = base_url if base_url is not None else get_mcp_base_url()

    embeddings_path = get_synthetic_embeddings_path()

    if not synthetic_embeddings_exist():
        if os.getenv("OPENBENCH_COPILOT_AUTOGEN", "1") in {"1", "true", "True"}:
            if os.getenv("OPENBENCH_COPILOT_SILENT", "1") not in {"1", "true", "True"}:
                print("Generating synthetic embeddings...")
            if not (os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")):
                raise RuntimeError("OPENAI_API_KEY is required to generate embeddings.")
            asyncio.run(generate_synthetic_embeddings())
        else:
            raise RuntimeError(
                f"Synthetic embeddings file not found at {embeddings_path}. "
                "Set OPENBENCH_COPILOT_AUTOGEN=1 to auto-generate, or run: "
                "python -m openbench.tools.progressivemcpbench.copilot.synthetic_arg_generation"
            )

    os.environ.setdefault("MCP_DATA_PATH", str(embeddings_path))

    @asynccontextmanager
    async def copilot_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        async with SyntheticCopilotRouter(
            mcp_arg_path=embeddings_path,
            base_url=resolved_base_url,
        ) as router:
            global _GLOBAL_ROUTER
            _GLOBAL_ROUTER = router
            try:
                yield {"router": router}
            finally:
                _GLOBAL_ROUTER = None

    server = FastMCP("synthetic-mcp-copilot", lifespan=copilot_lifespan)

    @server.tool(
        name="meta__route",
        description=(
            """
This is a tool used to find MCP servers and tools that can solve user needs    
    When to use this tool:
        -When faced with user needs, you (LLM) are unable to solve them on your own and do not have the tools to solve the problem.
        -When a user proposes a new task and you (LLM) are unsure which specific tool to use to complete it.
        -When the user's request is vague or complex, and feasible tool options need to be explored first.
        -This is the first step in executing unknown tasks, known as the "discovery" phase, aimed at finding the correct tool.
    **Parameter Description**
    Query (string, required): The input query must contain a <tool_assistant> tag with server and tool descriptions, for example: 
        <tool_assistant>
        server: ... # Platform/permission domain
        tool: ... # Operation type + target
        </tool_assistant>
"""
        ),
    )
    async def route(
        query: str,
        ctx: "Context | None" = None,
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
                    types.TextContent(type="text", text="Router context unavailable"),
                ],
            )
        try:
            result = await router.route(query)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result))]
            )
        except Exception as e:
            error_msg = f"Error routing query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=error_msg)],
            )

    @server.tool(
        name="meta__execute-tool",
        description=(
            """
A tool for executing a specific tool on a specific server. Select tools only from the results obtained from the previous meta__route each time.

When to use this tool:
    - When using the route tool to route to a specific MCP server and tool
    - When the 'execute-tool' fails to execute (up to 3 repetitions).
    - When the user's needs and previous needs require the same tool.

Parameters explained:
    -server_name: string, required. The name of the server where the target tool is located.

    -tool_name: string, required. The name of the target tool to be executed.

    -params: dictionary or None, optional. A dictionary containing all parameters that need to be passed to the target tool. This can be omitted if the target tool does not require parameters.
    
"""
        ),
    )
    async def execute_tool(
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None,
        ctx: "Context | None" = None,
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
        result = await router.call_tool(server_name, tool_name, params)
        return result

    server.run(transport="stdio")


if __name__ == "__main__":
    serve()
