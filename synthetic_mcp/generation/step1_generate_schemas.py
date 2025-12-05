#!/usr/bin/env python3
"""
Step 1: Generate Synthetic Tool Schemas

This script takes the server seeds from step0 and generates synthetic tool schemas
that will be used by the mock HTTP MCP server.

Key decisions:
- Preserve real server names and descriptions for realism
- Generate simplified tool schemas with handler specifications
- Each handler specifies how the mock server will respond (table_lookup, filesystem, etc.)
- Include ALL tools from each required server (for minimal-servers strategy)
- Tools that are explicitly required get proper handler implementations
- Other tools get stub handlers that return placeholder responses
"""

import json
from pathlib import Path
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
SEEDS_DIR = CONFIG_DIR / "seeds"
OUTPUT_FILE = CONFIG_DIR / "servers.json"


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON to file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {path}")


def get_tools_needed(working_tasks: list[dict]) -> dict[str, set[str]]:
    """Extract which tools are needed from each server."""
    tools_by_server: dict[str, set[str]] = {}
    for task in working_tasks:
        servers = task.get("required_servers", [])
        tools = task.get("required_tools", [])
        for server in servers:
            if server not in tools_by_server:
                tools_by_server[server] = set()
            for tool in tools:
                tools_by_server[server].add(tool)
    return tools_by_server


def get_required_servers(working_tasks: list[dict]) -> set[str]:
    """Get the set of all servers that are required by any task."""
    servers: set[str] = set()
    for task in working_tasks:
        for server in task.get("required_servers", []):
            servers.add(server)
    return servers


def create_stub_handler(tool_name: str, tool_info: dict) -> dict:
    """Create a stub handler for tools that are not explicitly required.
    
    Mutation tools return success messages.
    Query tools return empty/placeholder responses.
    """
    description = tool_info.get("description", "").lower()
    name_lower = tool_name.lower()
    
    # Detect mutation verbs in name or description
    mutation_verbs = [
        "create", "add", "delete", "remove", "update", "modify", "set",
        "write", "save", "copy", "move", "rename", "format", "protect",
        "unprotect", "replace", "convert", "customize"
    ]
    
    is_mutation = any(verb in name_lower or verb in description for verb in mutation_verbs)
    
    if is_mutation:
        return {
            "type": "static_json",
            "response": {
                "success": True,
                "message": f"Operation '{tool_name}' completed successfully (synthetic stub)"
            }
        }
    else:
        # Query tools - return placeholder based on what they might return
        if "list" in name_lower:
            return {"type": "static_json", "response": {"items": [], "message": "No items found (synthetic stub)"}}
        elif "get" in name_lower or "find" in name_lower or "search" in name_lower:
            return {"type": "static_json", "response": {"results": [], "message": "No results found (synthetic stub)"}}
        else:
            return {"type": "static_json", "response": {"data": None, "message": "No data available (synthetic stub)"}}


def create_fallback_handler(server_name: str, server_info: dict, needed_tools: set[str]) -> dict:
    """Create a fallback handler specification when LLM fails.
    
    Includes ALL tools from the server, with proper handlers for needed tools
    and stub handlers for non-needed tools.
    """
    original_tools = server_info.get("tools", [])
    
    # Default handler mappings based on server type
    handler_defaults = {
        "filesystem": {"type": "filesystem", "root": "/root"},
        "excel": {"type": "excel_reader", "root": "/root/excel"},
        "mcp-simple-arxiv": {"type": "table_lookup", "dataset": "data/api/arxiv_papers.json", "key_field": "paper_id"},
        "arxiv-mcp-server": {"type": "table_lookup", "dataset": "data/api/arxiv_papers.json", "key_field": "paper_id"},
        "biomcp": {"type": "table_lookup", "dataset": "data/api/clinical_trials.json", "key_field": "nct_id"},
        "music-analysis": {"type": "table_lookup", "dataset": "data/api/audio_analysis.json", "key_field": "file_path"},
        "maven-deps-server": {"type": "table_lookup", "dataset": "data/api/maven_releases.json", "key_field": "artifact"},
        "pdf-reader-mcp": {"type": "filesystem", "root": "/root"},
        "word-document-server": {"type": "filesystem", "root": "/root/word"},
        "searxng": {"type": "filesystem", "root": "/root"},
    }
    
    default_handler = handler_defaults.get(server_name, {"type": "static_json", "response": {}})
    
    tools_with_handlers = []
    for tool in original_tools:
        tool_name = tool.get("name", "")
        if tool_name in needed_tools:
            # Use the server's default handler for needed tools
            handler = default_handler.copy()
        else:
            # Use a stub handler for non-needed tools
            handler = create_stub_handler(tool_name, tool)
        
        tools_with_handlers.append({
            "name": tool_name,
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
            "handler": handler,
        })
    
    return {
        "server_name": server_name,
        "description": server_info.get("description", ""),
        "category": server_info.get("category", ""),
        "tools": tools_with_handlers,
    }


def main():
    print("=" * 60)
    print("Step 1: Generate Synthetic Tool Schemas")
    print("=" * 60)
    
    # Load seeds
    print("\n📂 Loading seeds...")
    servers_raw = load_json(SEEDS_DIR / "servers_raw.json")
    working_tasks = load_json(SEEDS_DIR / "working_tasks.json")
    print(f"  ✓ Loaded {len(servers_raw)} servers from seeds")
    print(f"  ✓ Loaded {len(working_tasks)} working tasks")
    
    # Load existing servers.json to preserve manually-defined servers
    existing_servers: dict = {}
    if OUTPUT_FILE.exists():
        existing_servers = load_json(OUTPUT_FILE)
        print(f"  ✓ Loaded {len(existing_servers)} servers from existing servers.json")
    
    # Get tools needed per server and required servers
    tools_needed = get_tools_needed(working_tasks)
    required_servers = get_required_servers(working_tasks)
    
    print("\n🔍 Required servers and tools needed:")
    for server in sorted(required_servers):
        tools = tools_needed.get(server, set())
        all_tools = len(servers_raw.get(server, {}).get("tools", []))
        source = "seeds" if server in servers_raw else ("existing" if server in existing_servers else "MISSING")
        print(f"  {server}: {len(tools)} needed / {all_tools} total tools [{source}]")
    
    # Generate schemas for each required server
    # We use fallback handler for all to include ALL tools from each server
    print("\n🔧 Generating tool schemas (including ALL tools from required servers)...")
    synthetic_servers = {}
    
    for server_name in sorted(required_servers):
        if server_name in servers_raw:
            # Generate from seeds with all tools
            server_info = servers_raw[server_name]
            needed = tools_needed.get(server_name, set())
            all_tools_count = len(server_info.get("tools", []))
            
            print(f"\n  📝 Processing {server_name}...")
            print(f"     Needed tools: {sorted(needed)}")
            print(f"     Total tools in server: {all_tools_count}")
            
            # Use fallback handler to include ALL tools
            result = create_fallback_handler(server_name, server_info, needed)
            synthetic_servers[server_name] = result
            
            needed_count = sum(1 for t in result.get("tools", []) if t.get("name") in needed)
            stub_count = len(result.get("tools", [])) - needed_count
            print(f"     ✓ Generated {needed_count} tools with handlers, {stub_count} with stubs")
        elif server_name in existing_servers:
            # Preserve manually-defined server from existing servers.json
            print(f"\n  📝 Preserving {server_name} from existing servers.json...")
            synthetic_servers[server_name] = existing_servers[server_name]
            tools_count = len(existing_servers[server_name].get("tools", []))
            print(f"     ✓ Preserved {tools_count} tools")
        else:
            print(f"\n  ⚠ Server '{server_name}' not found in seeds or existing servers.json, skipping")
    
    # Save output (sorted for reproducibility)
    print("\n💾 Saving output...")
    sorted_synthetic_servers = dict(sorted(synthetic_servers.items()))
    save_json(OUTPUT_FILE, sorted_synthetic_servers)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_tools = sum(len(s.get("tools", [])) for s in sorted_synthetic_servers.values())
    total_needed = sum(len(tools_needed.get(s, set())) for s in sorted_synthetic_servers.keys())
    print(f"  Servers generated: {len(sorted_synthetic_servers)}")
    print(f"  Total tools: {total_tools} ({total_needed} needed, {total_tools - total_needed} stubs)")
    
    for name, info in sorted_synthetic_servers.items():
        tools = info.get("tools", [])
        needed = tools_needed.get(name, set())
        handler_types = set(t.get("handler", {}).get("type", "unknown") for t in tools)
        print(f"    {name}: {len(tools)} tools, {len(needed)} needed ({', '.join(handler_types)})")
    
    print("\n✅ Step 1 complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Next: Review the schemas, then run step2_generate_data.py")


if __name__ == "__main__":
    main()
