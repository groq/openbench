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
- Tools that are explicitly required get proper handler implementations (via LLM if new)
- Other tools get stub handlers that return placeholder responses

Hybrid approach:
- Existing handlers in servers.json are preserved (idempotent)
- LLM is only invoked for NEW needed tools that don't have handlers
- Stub handlers are used for non-needed tools
"""

import json
import os
from pathlib import Path
from typing import Any

from groq import Groq

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
SEEDS_DIR = CONFIG_DIR / "seeds"
OUTPUT_FILE = CONFIG_DIR / "servers.json"

# Model configuration
MODEL = "openai/gpt-4.1"


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


def get_default_handler(server_name: str) -> dict:
    """Get the default handler for a server type."""
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
    return handler_defaults.get(server_name, {"type": "static_json", "response": {}})


def generate_handler_with_llm(
    client: Groq,
    server_name: str,
    server_info: dict,
    tool_info: dict,
    working_tasks: list[dict],
) -> dict | None:
    """Use LLM to generate a handler specification for a single tool."""
    
    # Find relevant tasks for context
    relevant_tasks = [
        t for t in working_tasks
        if server_name in t.get("required_servers", [])
        and tool_info.get("name") in t.get("required_tools", [])
    ]
    
    if not relevant_tasks:
        return None
    
    # Build task examples for context
    task_examples = [
        {"Question": t["Question"], "answer": t["answer"], "required_tools": t["required_tools"]}
        for t in relevant_tasks
    ]
    
    # Build the prompt
    prompt = f"""You are designing a synthetic MCP server handler for benchmarking LLM agents.

## Server: {server_name}
Description: {server_info.get('description', 'No description')}

## Tool to implement:
{json.dumps(tool_info, indent=2)}

## Tasks that will use this tool:
{json.dumps(task_examples, indent=2)}

## Your task:
Generate a handler specification that will allow this tool to return the expected answers.

Handler types available:
- "filesystem": For tools that read/list files. Specify "root" (the base directory).
- "table_lookup": For tools that look up data by key. Specify "dataset" (JSON file path) and "key_field".
- "static_json": For tools returning fixed data. Specify the "response" directly.
- "excel_reader": For Excel file operations. Specify "root" directory.
- "compute": For calculations. Specify "operation" type.
- "web_corpus": For web page tools. Specify "operation" (navigate, get_visible_html, screenshot).
- "url_search": For URL search tools.

Return ONLY a valid JSON object with the handler specification, for example:
{{"type": "filesystem", "root": "/root"}}
or
{{"type": "static_json", "response": {{"data": "example"}}}}

Return ONLY the JSON handler object, no markdown or explanations."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a technical architect designing mock MCP server handlers. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content
        if content is None:
            return None
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        return json.loads(content)
    except Exception as e:
        print(f"       ⚠ LLM generation failed: {e}")
        return None


def build_server_tools(
    server_name: str,
    server_info: dict,
    needed_tools: set[str],
    existing_tools: dict[str, dict],
    working_tasks: list[dict],
    client: Groq | None,
) -> tuple[list[dict], dict[str, int]]:
    """Build the tools list for a server, preserving existing handlers.
    
    Returns:
        tuple of (tools_list, stats_dict) where stats_dict has counts by source
    """
    original_tools = server_info.get("tools", [])
    default_handler = get_default_handler(server_name)
    
    tools_with_handlers = []
    stats = {"preserved": 0, "llm_generated": 0, "default": 0, "stub": 0}
    
    for tool in original_tools:
        tool_name = tool.get("name", "")
        
        # Check if we have an existing handler
        if tool_name in existing_tools:
            # Preserve existing handler
            handler = existing_tools[tool_name].get("handler", {})
            stats["preserved"] += 1
            source = "preserved"
        elif tool_name in needed_tools:
            # This is a needed tool without an existing handler - try LLM
            handler = None
            if client:
                handler = generate_handler_with_llm(
                    client, server_name, server_info, tool, working_tasks
                )
                if handler:
                    stats["llm_generated"] += 1
                    source = "llm"
            
            if not handler:
                # Fall back to default handler
                handler = default_handler.copy()
                stats["default"] += 1
                source = "default"
        else:
            # Non-needed tool - use stub
            handler = create_stub_handler(tool_name, tool)
            stats["stub"] += 1
            source = "stub"
        
        tools_with_handlers.append({
            "name": tool_name,
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
            "handler": handler,
        })
    
    return tools_with_handlers, stats


def main():
    print("=" * 60)
    print("Step 1: Generate Synthetic Tool Schemas")
    print("=" * 60)
    
    # Check for API key (optional - only needed for new tools)
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    if client:
        print("\n🔑 GROQ_API_KEY found - LLM available for new tool handlers")
    else:
        print("\n⚠ GROQ_API_KEY not set - will use default handlers for new tools")
    
    # Load seeds
    print("\n📂 Loading seeds...")
    servers_raw = load_json(SEEDS_DIR / "servers_raw.json")
    working_tasks = load_json(SEEDS_DIR / "working_tasks.json")
    print(f"  ✓ Loaded {len(servers_raw)} servers from seeds")
    print(f"  ✓ Loaded {len(working_tasks)} working tasks")
    
    # Load existing servers.json to preserve handlers
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
    print("\n🔧 Generating tool schemas (including ALL tools from required servers)...")
    synthetic_servers = {}
    total_stats = {"preserved": 0, "llm_generated": 0, "default": 0, "stub": 0}
    
    for server_name in sorted(required_servers):
        if server_name in servers_raw:
            # Generate from seeds with all tools
            server_info = servers_raw[server_name]
            needed = tools_needed.get(server_name, set())
            
            # Build index of existing tools for this server
            existing_tools: dict[str, dict] = {}
            if server_name in existing_servers:
                for tool in existing_servers[server_name].get("tools", []):
                    existing_tools[tool.get("name", "")] = tool
            
            print(f"\n  📝 Processing {server_name}...")
            print(f"     Needed tools: {sorted(needed)}")
            print(f"     Existing handlers: {len(existing_tools)}")
            
            tools_list, stats = build_server_tools(
                server_name, server_info, needed, existing_tools, working_tasks, client
            )
            
            synthetic_servers[server_name] = {
                "server_name": server_name,
                "description": server_info.get("description", ""),
                "category": server_info.get("category", ""),
                "tools": tools_list,
            }
            
            # Update totals
            for key in total_stats:
                total_stats[key] += stats[key]
            
            print(f"     ✓ {len(tools_list)} tools: {stats['preserved']} preserved, "
                  f"{stats['llm_generated']} LLM, {stats['default']} default, {stats['stub']} stub")
            
        elif server_name in existing_servers:
            # Preserve manually-defined server from existing servers.json
            print(f"\n  📝 Preserving {server_name} from existing servers.json...")
            synthetic_servers[server_name] = existing_servers[server_name]
            tools_count = len(existing_servers[server_name].get("tools", []))
            total_stats["preserved"] += tools_count
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
    print(f"  Servers: {len(sorted_synthetic_servers)}")
    print(f"  Total tools: {total_tools}")
    print(f"    - Preserved: {total_stats['preserved']}")
    print(f"    - LLM generated: {total_stats['llm_generated']}")
    print(f"    - Default handler: {total_stats['default']}")
    print(f"    - Stub: {total_stats['stub']}")
    
    print("\n  By server:")
    for name, info in sorted_synthetic_servers.items():
        tools = info.get("tools", [])
        needed = tools_needed.get(name, set())
        handler_types = set(t.get("handler", {}).get("type", "unknown") for t in tools)
        print(f"    {name}: {len(tools)} tools, {len(needed)} needed ({', '.join(sorted(handler_types))})")
    
    print("\n✅ Step 1 complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Next: Review the schemas, then run step2_generate_data.py")


if __name__ == "__main__":
    main()
