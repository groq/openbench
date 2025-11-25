"""
Helpers for loading MCP server configs used by ProgressiveMCPBench.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openbench.tools.progressivemcpbench.copilot.schemas import Server, ServerConfig
from openbench.tools.progressivemcpbench.common.paths import root_sandbox_dir
from openbench.tools.progressivemcpbench.copilot.upstream_cache import (
    get_clean_config_cached,
)

logger = logging.getLogger(__name__)


def load_servers_from_config(
    config: dict[str, Any] | Path | None = None,
) -> dict[str, Server]:
    """Load server definitions from clean_config with sandbox path adjustments."""
    servers: dict[str, Server] = {}

    if config is None:
        config_obj, _ = get_clean_config_cached()
    elif isinstance(config, dict):
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

    sandbox = root_sandbox_dir()
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
                parts = a.split("annotated_data/", 1)
                if len(parts) == 2:
                    args[i] = str(sandbox / parts[1])
        cfg["args"] = args
        servers[name] = Server(name=name, config=ServerConfig(**cfg))

    return servers
