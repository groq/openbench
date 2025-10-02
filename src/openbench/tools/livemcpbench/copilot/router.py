"""
Router for semantic tool selection and tool execution.

Credit: Adapted from LiveMCPBench baseline router:
https://github.com/icip-cas/LiveMCPBench/blob/main/baseline/mcp_copilot/router.py

"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .matcher import ToolMatcher
from .schemas import Server, ServerConfig
from .mcp_connection import MCPConnection


import mcp.types as types

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[0]


def _root_sandbox_dir() -> Path:
    return Path(os.path.expanduser("~/.openbench/livemcpbench/root")).resolve()


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
            # Ensure parent dir exists for prospective outputs
            p = Path(new_path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return new_path
    return obj


def dump_to_yaml(data: dict[str, Any]) -> str:
    """Serialize a dict to YAML (fallback to JSON if PyYAML is unavailable)."""
    if yaml is not None:
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    # Fallback
    return json.dumps(data, indent=2, ensure_ascii=False)


class Router:
    _default_config_path = PROJECT_ROOT / "config" / "clean_config.json"

    def __init__(self, config: dict[str, Any] | Path = _default_config_path):
        self.servers: dict[str, Server] = {}

        # Load config
        if isinstance(config, dict):
            config_obj = config
        elif isinstance(config, Path):
            if config.exists():
                with config.open("r") as f:
                    config_obj = json.load(f)
            else:
                logger.warning(
                    f"Config file not found at {config}. Starting with empty server list."
                )
                config_obj = {"mcpServers": {}}
        else:
            raise ValueError("Config must be a dictionary or a Path to a JSON file.")

        # Overlay paths to use root sandbox
        sandbox = _root_sandbox_dir()
        mcp_servers = config_obj.get("mcpServers", {})
        for name, config_data in mcp_servers.items():
            cfg = dict(config_data)
            args = list(cfg.get("args", []))
            # Redirect filesystem base path
            if name.lower() in {
                "filesystem",
                "server-filesystem",
                "@modelcontextprotocol/server-filesystem",
            }:
                if args and args[-1] == "/":
                    args[-1] = str(sandbox)
            # Replace annotated_data relative paths
            for i, a in enumerate(args):
                if isinstance(a, str) and "annotated_data" in a:
                    # Map './annotated_data/...' -> sandbox/<...>
                    parts = a.split("annotated_data/", 1)
                    if len(parts) == 2:
                        args[i] = str(sandbox / parts[1])
            cfg["args"] = args
            self.servers[name] = Server(name=name, config=ServerConfig(**cfg))

        # Initialize ToolMatcher
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_dims = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        top_servers = int(os.getenv("TOP_SERVERS", "5"))
        top_tools = int(os.getenv("TOP_TOOLS", "3"))

        self.matcher = ToolMatcher(
            embedding_model=embedding_model,  # type: ignore[arg-type]
            dimensions=embedding_dims,
            top_servers=top_servers,
            top_tools=top_tools,
        )

        # Setup OpenAI client + load precomputed embeddings file
        base_url = os.getenv("EMBEDDING_BASE_URL")  # default OpenAI when None
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

        # Default to packaged path but allow override via MCP_DATA_PATH
        default_data_path = (
            PROJECT_ROOT
            / "config"
            / f"mcp_arg_{os.getenv('EMBEDDING_MODEL')}_{os.getenv('ABSTRACT_MODEL')}.json"
        )
        data_path_env = os.getenv("MCP_DATA_PATH")
        data_path = Path(data_path_env) if data_path_env else default_data_path

        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or EMBEDDING_API_KEY for routing embeddings."
            )
        if not data_path.exists():
            raise ValueError(f"MCP data file not found at: {data_path}")

        self.matcher.setup_openai_client(base_url=base_url, api_key=api_key)  # type: ignore[arg-type]
        self.matcher.load_data(str(data_path))

        # Synchronize connection lifecycle
        self.connection_lock = asyncio.Lock()

    async def route(self, query: str) -> dict[str, Any]:
        """Route using ToolMatcher to find the best tools."""
        return self.matcher.match(query)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> types.CallToolResult:
        """Execute a tool on a given server with a fresh connection."""

        async with self.connection_lock:
            server_config = self.servers.get(server_name)
            if not server_config:
                raise ValueError(
                    f"Server '{server_name}' is not defined in the configuration."
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

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
