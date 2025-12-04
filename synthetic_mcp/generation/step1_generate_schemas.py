#!/usr/bin/env python3
"""
Step 1: Generate Synthetic Tool Schemas

This script takes the server seeds from step0 and generates synthetic tool schemas
that will be used by the mock HTTP MCP server. For the MVP, we focus on the tools
actually used by the 13 working tasks.

Key decisions:
- Preserve real server names and descriptions for realism
- Generate simplified tool schemas with handler specifications
- Each handler specifies how the mock server will respond (table_lookup, filesystem, etc.)
- Include the exact tools needed for working tasks + a few extras for distraction strategies

Uses: gpt-oss-120b via Groq API
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
MODEL = "openai/gpt-oss-120b"


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


def generate_handler_spec(
    client: Groq,
    server_name: str,
    server_info: dict,
    needed_tools: set[str],
    working_tasks: list[dict],
) -> dict:
    """Use LLM to generate handler specifications for a server's tools."""
    
    # Get the original tools that are needed
    original_tools = server_info.get("tools", [])
    needed_tool_list = [t for t in original_tools if t.get("name") in needed_tools]
    
    # Find relevant tasks for context
    relevant_tasks = [
        t for t in working_tasks
        if server_name in t.get("required_servers", [])
    ]
    
    # Build task examples for context
    task_examples = [
        {"Question": t["Question"], "answer": t["answer"], "required_tools": t["required_tools"]}
        for t in relevant_tasks
    ]
    
    # Build the prompt
    prompt = f"""You are designing a synthetic MCP server that will be used for benchmarking LLM agents.

## Server: {server_name}
Description: {server_info.get('description', 'No description')}
Category: {server_info.get('category', 'Unknown')}

## Tools needed for benchmark tasks:
{json.dumps(needed_tool_list, indent=2)}

## Example tasks that will use these tools:
{json.dumps(task_examples, indent=2)}

## Your task:
Generate a synthetic tool specification for each tool. For each tool, provide:
1. Keep the original name, description, and inputSchema exactly as-is
2. Add a "handler" object that specifies how a mock HTTP server should respond

Handler types available:
- "filesystem": For tools that read/list files. Specify "root" (the base directory).
- "table_lookup": For tools that look up data by key. Specify "dataset" (JSON file path) and "key_field".
- "static_json": For tools returning fixed data. Specify the "response" directly.
- "excel_reader": For Excel file operations. Specify "root" directory.
- "compute": For calculations. Specify "operation" type.

Important: The handler must be able to produce the expected answers shown in the tasks.

Return ONLY a valid JSON object with this structure:
{{
  "server_name": "{server_name}",
  "description": "<original description>",
  "category": "<original category>",
  "tools": [
    {{
      "name": "<tool name>",
      "description": "<original description>",
      "inputSchema": <original schema>,
      "handler": {{
        "type": "<handler type>",
        // handler-specific fields
      }}
    }}
  ]
}}

Return ONLY the JSON, no markdown code blocks or explanations."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a technical architect designing mock MCP servers for AI benchmarking. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Try to parse the JSON response
    try:
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ⚠ Failed to parse LLM response for {server_name}: {e}")
        print(f"  Response: {content[:500]}...")
        return None


def create_fallback_handler(server_name: str, server_info: dict, needed_tools: set[str]) -> dict:
    """Create a fallback handler specification when LLM fails."""
    original_tools = server_info.get("tools", [])
    needed_tool_list = [t for t in original_tools if t.get("name") in needed_tools]
    
    # Default handler mappings based on server type
    handler_defaults = {
        "filesystem": {"type": "filesystem", "root": "data/files/root"},
        "excel": {"type": "excel_reader", "root": "data/excel"},
        "mcp-simple-arxiv": {"type": "table_lookup", "dataset": "data/api/arxiv_papers.json", "key_field": "paper_id"},
        "arxiv-mcp-server": {"type": "table_lookup", "dataset": "data/api/arxiv_papers.json", "key_field": "paper_id"},
        "biomcp": {"type": "table_lookup", "dataset": "data/api/clinical_trials.json", "key_field": "nct_id"},
        "music-analysis": {"type": "table_lookup", "dataset": "data/api/audio_analysis.json", "key_field": "file_path"},
        "maven-deps-server": {"type": "table_lookup", "dataset": "data/api/maven_releases.json", "key_field": "artifact"},
        "pdf-reader-mcp": {"type": "filesystem", "root": "data/files/root"},
        "word-document-server": {"type": "filesystem", "root": "data/files/root"},
        "searxng": {"type": "filesystem", "root": "data/files/root"},
    }
    
    default_handler = handler_defaults.get(server_name, {"type": "static_json", "response": {}})
    
    tools_with_handlers = []
    for tool in needed_tool_list:
        tools_with_handlers.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
            "handler": default_handler.copy(),
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
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required")
    
    client = Groq(api_key=api_key)
    
    # Load seeds
    print("\n📂 Loading seeds...")
    servers_raw = load_json(SEEDS_DIR / "servers_raw.json")
    working_tasks = load_json(SEEDS_DIR / "working_tasks.json")
    print(f"  ✓ Loaded {len(servers_raw)} servers")
    print(f"  ✓ Loaded {len(working_tasks)} working tasks")
    
    # Get tools needed per server
    tools_needed = get_tools_needed(working_tasks)
    print("\n🔍 Tools needed per server:")
    for server, tools in sorted(tools_needed.items()):
        print(f"  {server}: {sorted(tools)}")
    
    # Generate schemas for each server
    print("\n🤖 Generating synthetic schemas with gpt-oss-120b...")
    synthetic_servers = {}
    
    for server_name, server_info in servers_raw.items():
        needed = tools_needed.get(server_name, set())
        if not needed:
            print(f"  ⏭ Skipping {server_name} (no tools needed)")
            continue
        
        print(f"\n  📝 Processing {server_name}...")
        print(f"     Tools needed: {sorted(needed)}")
        
        result = generate_handler_spec(
            client,
            server_name,
            server_info,
            needed,
            working_tasks,
        )
        
        if result:
            synthetic_servers[server_name] = result
            print(f"     ✓ Generated {len(result.get('tools', []))} tools with handlers")
        else:
            print(f"     ⚠ LLM generation failed, using fallback")
            result = create_fallback_handler(server_name, server_info, needed)
            synthetic_servers[server_name] = result
            print(f"     ✓ Created fallback with {len(result.get('tools', []))} tools")
    
    # Save output
    print("\n💾 Saving output...")
    save_json(OUTPUT_FILE, synthetic_servers)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_tools = sum(len(s.get("tools", [])) for s in synthetic_servers.values())
    print(f"  Servers generated: {len(synthetic_servers)}")
    print(f"  Total tools: {total_tools}")
    
    for name, info in synthetic_servers.items():
        tools = info.get("tools", [])
        handler_types = set(t.get("handler", {}).get("type", "unknown") for t in tools)
        print(f"    {name}: {len(tools)} tools ({', '.join(handler_types)})")
    
    print("\n✅ Step 1 complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Next: Review the schemas, then run step2_generate_data.py")


if __name__ == "__main__":
    main()
