"""
MCP Copilot server entrypoint (OpenBench port).

Credit: Based on LiveMCPBench baseline server:
https://github.com/icip-cas/LiveMCPBench/blob/main/baseline/mcp_copilot/server.py

"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import asyncio
import logging
import os
import json

import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP

from .router import Router, dump_to_yaml
from .recorder import Recorder

# Reuse OpenBench arg generator to build per-server embeddings
from .arg_generation import McpArgGenerator
from .upstream_cache import (
    get_tools_json_cached,
    get_clean_config_cached,
)

logger = logging.getLogger(__name__)

_RECORD_DB = os.getenv("PROGRESSIVE_MCP_DB")
_RECORD_MODE = os.getenv("PROGRESSIVE_MCP_RECORD", "0") in {"1", "true", "True"}
_STEP_INDEX = 0
_RECORDER = None

def _get_recorder():
    global _RECORDER
    if _RECORDER is None and _RECORD_MODE and _RECORD_DB:
        task_id = os.getenv("PROGRESSIVE_MCP_TASK_ID", "unknown")
        model = os.getenv("PROGRESSIVE_MCP_MODEL", "unknown")
        _RECORDER = Recorder(Path(_RECORD_DB), task_id, model)
    return _RECORDER


def _configure_logging():
    """Reduce noisy logs from Copilot and underlying libraries by default."""
    silent = os.getenv("OPENBENCH_COPILOT_SILENT", "1") in {"1", "true", "True"}
    if not silent:
        return
    # Set root to WARNING to suppress INFO/DEBUG
    logging.basicConfig(level=logging.WARNING)
    # Quiet common noisy loggers
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
_GLOBAL_ROUTER: "Router | None" = None


def _user_cache_dir() -> Path:
    # Avoid introducing a hard dependency on platformdirs
    # Migrate caches to ~/.openbench
    return Path(os.path.expanduser("~/.openbench/livemcpbench/copilot")).resolve()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_clean_config_from_cache() -> dict[str, Any]:
    """Load curated clean_config.json from cache (fetch if needed)."""
    config, _ = get_clean_config_cached()
    return config


async def _generate_embeddings_file(output_file: Path) -> None:
    """Generate the per-server embedding file using upstream tools data.

    This mirrors baseline arg_generation.py behavior but feeds the config list
    directly (written to a temp JSON file to match generator's expectations).
    """
    # Ensure tools.json is present in cache (fetch if needed)
    _, cached_tools_path = get_tools_json_cached()

    # Run generator on cached tools.json
    generator = McpArgGenerator(config=cached_tools_path, output_file=output_file)
    await generator.generate()


def serve(config: dict[str, Any] | Path | None = None) -> None:
    """Run the Copilot MCP server (stdio).

    Args:
        config: Optional MCP server config dict or path to JSON with mcpServers.
                If None, fetches curated clean_config.json from upstream.
    """
    # Configure logging first
    _configure_logging()

    # Ensure embeddings file exists in user cache based on env models
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4.1-2025-04-14")

    output_dir = _user_cache_dir() / "config"
    _ensure_parent_dir(output_dir)
    
    # Allow override via MCP_DATA_PATH, else default to cache
    env_data_path = os.getenv("MCP_DATA_PATH")
    if env_data_path:
        mcp_arg_path = Path(env_data_path)
        # Ensure parent exists
        if not mcp_arg_path.parent.exists():
             mcp_arg_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        mcp_arg_path = output_dir / f"mcp_arg_{embedding_model}_{abstract_model}.json"
        # Point Router to this path
        os.environ["MCP_DATA_PATH"] = str(mcp_arg_path)

    # Generate embeddings file if missing
    if not mcp_arg_path.exists():
        if os.getenv("OPENBENCH_COPILOT_AUTOGEN", "0") in {"1", "true", "True"}:
            if os.getenv("OPENBENCH_COPILOT_SILENT", "1") not in {"1", "true", "True"}:
                print("Indexing MCP servers and tools...")
            if not (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("EMBEDDING_API_KEY")
                or os.getenv("ABSTRACT_API_KEY")
            ):
                # Allow proceeding without keys (dummy embeddings) if forced
                if os.getenv("OPENBENCH_COPILOT_NO_EMBEDDINGS", "0") not in {"1", "true", "True"}:
                    raise RuntimeError(
                        "OPENAI_API_KEY is required to generate embeddings (or provide EMBEDDING_API_KEY/ABSTRACT_API_KEY)."
                    )
            
            config_path_env = os.getenv("MCP_SERVERS_CONFIG")
            if config_path_env and Path(config_path_env).exists():
                generator = McpArgGenerator(config=Path(config_path_env), output_file=mcp_arg_path)
                asyncio.run(generator.generate())
            else:
                asyncio.run(_generate_embeddings_file(mcp_arg_path))
        else:
            raise RuntimeError(
                "Copilot embeddings file not found. Ensure the embeddings cache exists (it's auto-prepared by 'openbench eval livemcpbench'), "
                "or set MCP_DATA_PATH to an existing mcp_arg_*.json. You can validate the cache with 'openbench cache info'."
            )

    # Prepare Router config
    config_path_env = os.getenv("MCP_SERVERS_CONFIG")
    if config is None:
        if config_path_env:
             with open(config_path_env) as f:
                 config = json.load(f)
        else:
            try:
                config = _load_clean_config_from_cache()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load curated clean_config.json (online or cached): {e}"
                )

    @asynccontextmanager
    async def copilot_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        async with Router(config) as router:  # type: ignore[arg-type]
            global _GLOBAL_ROUTER
            _GLOBAL_ROUTER = router
            try:
                yield {"router": router}
            finally:
                _GLOBAL_ROUTER = None

    server = FastMCP("mcp-copilot", lifespan=copilot_lifespan)

    @server.tool(
        name="route",
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
            
            if _RECORD_MODE and _RECORD_DB:
                task_id = os.getenv("PROGRESSIVE_MCP_TASK_ID", "unknown")
                model = os.getenv("PROGRESSIVE_MCP_MODEL", "unknown")
                result_dict = {"content": [{"type": "text", "text": dump_to_yaml(result)}]}
                
                global _STEP_INDEX
                rec = _get_recorder()
                if rec:
                    rec.log_tool_call(
                        step_index=_STEP_INDEX,
                        call_type="route",
                        request={"query": query},
                        result=result_dict,
                    )
                _STEP_INDEX += 1

            return types.CallToolResult(
                content=[types.TextContent(type="text", text=dump_to_yaml(result))]
            )
        except Exception as e:
            error_msg = f"Error routing query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=error_msg)],
            )

    @server.tool(
        name="execute-tool",
        description=(
            """
A tool for executing a specific tool on a specific server.Select tools only from the results obtained from the previous route each time.

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
        
        if _RECORD_MODE and _RECORD_DB:
            task_id = os.getenv("PROGRESSIVE_MCP_TASK_ID", "unknown")
            model = os.getenv("PROGRESSIVE_MCP_MODEL", "unknown")
            
            # result is CallToolResult, verify if model_dump is available (pydantic model)
            # mcp.types.CallToolResult is a Pydantic model
            result_dict = result.model_dump()
            
            global _STEP_INDEX
            rec = _get_recorder()
            if rec:
                rec.log_tool_call(
                    step_index=_STEP_INDEX,
                    call_type="execute",
                    request={
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "params": params or {},
                    },
                    result=result_dict,
                )
            _STEP_INDEX += 1
            
        return result

    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    serve()
