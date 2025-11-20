import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

async def discover_tools_for_server(name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    server_params = StdioServerParameters(
        command=config["command"],
        args=config.get("args", []),
        env={**os.environ, **config.get("env", {})},
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [tool.model_dump() for tool in result.tools]

async def generate_tools_json(servers_config_path: Path, output_path: Path):
    with open(servers_config_path) as f:
        config = json.load(f)
    
    mcp_servers = config.get("mcpServers", {})
    output_list = []
    
    for name, server_config in mcp_servers.items():
        print(f"Discovering tools for {name}...")
        try:
            tools = await discover_tools_for_server(name, server_config)
            
            # Construct the entry in the format expected by McpArgGenerator
            entry = {
                "config": {
                    "mcpServers": {
                        name: server_config
                    }
                },
                "tools": {
                    name: {
                        "tools": tools
                    }
                },
                "description": f"MCP server for {name}" 
            }
            output_list.append(entry)
        except Exception as e:
            print(f"Error discovering tools for {name}: {e}")
            
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2)
    
    print(f"Tools discovery complete. Saved to {output_path}")

if __name__ == "__main__":
    # Test usage
    import sys
    if len(sys.argv) > 2:
        asyncio.run(generate_tools_json(Path(sys.argv[1]), Path(sys.argv[2])))
