import json
import asyncio
import os
from pathlib import Path
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def get_tools_from_server(config):
    env = os.environ.copy()
    if config.get("env"):
        env.update(config["env"])
    
    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools

async def generate_tools_json(servers_config_path: Path, output_path: Path):
    with open(servers_config_path) as f:
        config = json.load(f)
    
    tools_data = []
    
    mcp_servers = config.get("mcpServers", {})
    
    for name, server_config in mcp_servers.items():
        try:
            print(f"Discovering tools for {name}...")
            tools = await get_tools_from_server(server_config)
            
            server_info = {
                "server_name": name,
                "server_description": f"MCP server for {name}", # Placeholder
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameter": tool.inputSchema
                    }
                    for tool in tools
                ]
            }
            tools_data.append(server_info)
        except Exception as e:
            print(f"Failed to get tools for {name}: {e}")
            
    with open(output_path, "w") as f:
        json.dump(tools_data, f, indent=2)
    
    print(f"Saved tools to {output_path}")
