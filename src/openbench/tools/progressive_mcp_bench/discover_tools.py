import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from openbench.tools.livemcpbench.copilot.mcp_connection import MCPConnection
from openbench.tools.livemcpbench.copilot.schemas import Server, ServerConfig

async def generate_tools_json(servers_config_path: Path, output_path: Path) -> None:
    with open(servers_config_path, "r") as f:
        config = json.load(f)
        
    mcp_servers = config.get("mcpServers", {})
    servers_data = []
    
    for name, config_data in mcp_servers.items():
        print(f"Discovering tools for {name}...")
        cfg = dict(config_data)
        server_config = ServerConfig(**cfg)
        server = Server(name=name, config=server_config)
        
        try:
            async with MCPConnection(server) as conn:
                tools = await conn.list_tools()
                # Format compatible with McpArgGenerator
                # It expects list of objects like:
                # {
                #   "config": { "mcpServers": { "name": { ... } } },
                #   "description": "...",
                #   "tools": { "name": { "tools": [ ... ] } }
                # }
                # This is a bit weird structure from upstream, let's replicate it.
                
                server_info = {
                    "config": {
                        "mcpServers": {
                            name: config_data
                        }
                    },
                    "description": f"MCP server for {name}",
                    "tools": {
                        name: {
                            "tools": [t.model_dump() for t in tools]
                        }
                    }
                }
                
                servers_data.append(server_info)
        except Exception as e:
            print(f"Failed to connect to {name}: {e}")
            import traceback
            traceback.print_exc()
            
    with open(output_path, "w") as f:
        json.dump(servers_data, f, indent=2)
        
    print(f"Saved tools to {output_path}")
