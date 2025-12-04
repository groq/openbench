#!/usr/bin/env python3
"""
Step 0: Extract Seeds from Working Tasks

This script extracts the MCP servers and tools used by the working tasks
in ProgressiveMCPBench (those with non-null answers and required_servers).

It reads:
- progressivemcpbench.json: The current dataset with working tasks
- tools.json: The full tool definitions from LiveMCPBench
- clean_config.json: The MCP server configurations

It outputs:
- seeds/servers_raw.json: Server names, descriptions, and tool schemas for the MVP
- seeds/working_tasks.json: The working tasks with their tool requirements
"""

import json
from pathlib import Path
from typing import Any

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "src/openbench/datasets/data/progressivemcpbench.json"
CACHE_DIR = Path.home() / ".openbench/progressivemcpbench/copilot/raw"
TOOLS_JSON_PATH = CACHE_DIR / "tools.json"
CLEAN_CONFIG_PATH = CACHE_DIR / "clean_config.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "config" / "seeds"


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


def extract_working_tasks(dataset: list[dict]) -> list[dict]:
    """Extract tasks that have answers (non-null) and required_servers."""
    working = []
    for task in dataset:
        # Skip tasks with null answers or underscore-prefixed answers (broken)
        answer = task.get("answer")
        if answer is None:
            continue
        
        # Must have required_servers to be in MVP
        required_servers = task.get("required_servers")
        if not required_servers:
            continue
        
        working.append({
            "task_id": task["task_id"],
            "Question": task["Question"],
            "category": task.get("category"),
            "file_name": task.get("file_name"),
            "answer": answer,
            "scorer_instructions": task.get("scorer_instructions"),
            "required_servers": required_servers,
            "required_tools": task.get("required_tools", []),
            "annotator_metadata": task.get("Annotator Metadata", {}),
        })
    
    return working


def build_server_index(tools_json: list[dict]) -> dict[str, dict]:
    """Build an index of server name -> server info from tools.json."""
    index = {}
    for server_entry in tools_json:
        # tools.json has a nested structure: tools -> server_name -> tools list
        tools_data = server_entry.get("tools", {})
        for server_name, server_info in tools_data.items():
            if server_name not in index:
                index[server_name] = {
                    "name": server_entry.get("name", server_name),
                    "description": server_entry.get("description", ""),
                    "category": server_entry.get("category", ""),
                    "web": server_entry.get("web", ""),
                    "tools": server_info.get("tools", []),
                }
    return index


def extract_server_seeds(
    working_tasks: list[dict],
    server_index: dict[str, dict],
    clean_config: dict,
) -> dict[str, dict]:
    """Extract server seeds for servers used in working tasks."""
    required_server_names = set()
    for task in working_tasks:
        for server in task.get("required_servers", []):
            required_server_names.add(server)
    
    print(f"\n  Required servers from working tasks: {sorted(required_server_names)}")
    
    seeds = {}
    mcp_servers = clean_config.get("mcpServers", {})
    
    for server_name in required_server_names:
        # Get info from tools.json index
        server_info = server_index.get(server_name, {})
        
        # Check if server exists in clean_config
        in_config = server_name in mcp_servers
        
        seeds[server_name] = {
            "name": server_info.get("name", server_name),
            "description": server_info.get("description", f"MCP server: {server_name}"),
            "category": server_info.get("category", "Unknown"),
            "web": server_info.get("web", ""),
            "in_clean_config": in_config,
            "tools": server_info.get("tools", []),
        }
        
        if not server_info:
            print(f"  ⚠ Server '{server_name}' not found in tools.json")
        elif not server_info.get("tools"):
            print(f"  ⚠ Server '{server_name}' has no tools in tools.json")
    
    return seeds


def main():
    print("=" * 60)
    print("Step 0: Extract Seeds from Working Tasks")
    print("=" * 60)
    
    # Load input files
    print("\n📂 Loading input files...")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    dataset = load_json(DATASET_PATH)
    print(f"  ✓ Loaded dataset: {len(dataset)} tasks")
    
    if not TOOLS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"tools.json not found: {TOOLS_JSON_PATH}\n"
            "Run the ProgressiveMCPBench eval first to populate the cache."
        )
    tools_json = load_json(TOOLS_JSON_PATH)
    print(f"  ✓ Loaded tools.json: {len(tools_json)} server entries")
    
    if not CLEAN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"clean_config.json not found: {CLEAN_CONFIG_PATH}")
    clean_config = load_json(CLEAN_CONFIG_PATH)
    print(f"  ✓ Loaded clean_config.json: {len(clean_config.get('mcpServers', {}))} servers")
    
    # Extract working tasks
    print("\n🔍 Extracting working tasks...")
    working_tasks = extract_working_tasks(dataset)
    print(f"  Found {len(working_tasks)} working tasks with required_servers")
    
    for task in working_tasks:
        print(f"    - {task['task_id'][:8]}... : {task['required_servers']} → {task['required_tools']}")
    
    # Build server index
    print("\n🗂 Building server index from tools.json...")
    server_index = build_server_index(tools_json)
    print(f"  Indexed {len(server_index)} servers")
    
    # Extract server seeds
    print("\n🌱 Extracting server seeds...")
    server_seeds = extract_server_seeds(working_tasks, server_index, clean_config)
    
    # Save outputs (sorted for reproducibility)
    print("\n💾 Saving outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sorted_server_seeds = dict(sorted(server_seeds.items()))
    save_json(OUTPUT_DIR / "servers_raw.json", sorted_server_seeds)
    save_json(OUTPUT_DIR / "working_tasks.json", working_tasks)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Working tasks: {len(working_tasks)}")
    print(f"  Required servers: {len(sorted_server_seeds)}")
    for name, info in sorted_server_seeds.items():
        tool_count = len(info.get("tools", []))
        status = "✓" if info.get("in_clean_config") else "✗"
        print(f"    [{status}] {name}: {tool_count} tools - {info.get('description', '')[:50]}...")
    
    print("\n✅ Step 0 complete!")
    print(f"   Next: Review {OUTPUT_DIR / 'servers_raw.json'}")
    print(f"         Then run step1_generate_schemas.py")


if __name__ == "__main__":
    main()
