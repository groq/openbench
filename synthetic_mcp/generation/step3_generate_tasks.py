#!/usr/bin/env python3
"""
Step 3: Generate Synthetic Tasks

This script uses the 13 working tasks from LiveMCPBench as the task dataset.
Since these tasks already have verified answers and required_servers/tools,
we use them directly rather than generating new synthetic tasks.

The key insight is that the synthetic MCP server will return deterministic
responses that match the expected answers - we're testing the LLM's ability
to use tools correctly, not the MCP server's functionality.

This step:
1. Loads the working_tasks.json from step0
2. Validates each task has the required fields
3. Cross-references with servers.json to ensure all tools exist
4. Outputs the final task dataset for the synthetic benchmark
"""

import json
from pathlib import Path
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_MCP_DIR = SCRIPT_DIR.parent
CONFIG_DIR = SYNTHETIC_MCP_DIR / "config"
SEEDS_DIR = CONFIG_DIR / "seeds"
TASKS_DIR = SYNTHETIC_MCP_DIR / "tasks"
DATA_DIR = SYNTHETIC_MCP_DIR / "data"


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


def get_all_tools_from_servers(servers: dict) -> dict[str, set[str]]:
    """Extract all tool names from servers.json, organized by server."""
    tools_by_server: dict[str, set[str]] = {}
    for server_name, server_info in servers.items():
        tools = server_info.get("tools", [])
        tools_by_server[server_name] = {t.get("name") for t in tools}
    return tools_by_server


def validate_task(task: dict, tools_by_server: dict[str, set[str]]) -> list[str]:
    """Validate a task has all required fields and tools exist. Returns list of issues."""
    issues = []

    # Required fields
    required_fields = ["task_id", "Question", "answer", "required_servers", "required_tools"]
    for field in required_fields:
        if not task.get(field):
            issues.append(f"Missing or empty field: {field}")

    # Check servers exist
    for server in task.get("required_servers", []):
        if server not in tools_by_server:
            issues.append(f"Unknown server: {server}")

    # Check tools exist in their servers
    for tool in task.get("required_tools", []):
        found = False
        for server in task.get("required_servers", []):
            if server in tools_by_server and tool in tools_by_server[server]:
                found = True
                break
        if not found:
            # Tool might be in a different server - just warn
            all_tools = set()
            for tools in tools_by_server.values():
                all_tools.update(tools)
            if tool not in all_tools:
                issues.append(f"Tool not found in any server: {tool}")

    return issues


def enrich_task_with_metadata(task: dict, data_dir: Path) -> dict:
    """Add additional metadata to task for the synthetic benchmark."""
    enriched = task.copy()

    # Add file_path mappings for tasks that reference files
    file_name = task.get("file_name", "")
    if file_name:
        # Map /root/... paths to synthetic data paths
        enriched["synthetic_file_path"] = str(data_dir / "files" / "root" / file_name.lstrip("/root/"))

    return enriched


def main():
    print("=" * 60)
    print("Step 3: Generate Synthetic Tasks")
    print("=" * 60)

    # Load inputs
    print("\n📂 Loading inputs...")

    working_tasks_path = SEEDS_DIR / "working_tasks.json"
    if not working_tasks_path.exists():
        raise FileNotFoundError(f"Working tasks not found: {working_tasks_path}\nRun step0 first.")

    working_tasks = load_json(working_tasks_path)
    print(f"  ✓ Loaded {len(working_tasks)} working tasks")

    servers_path = CONFIG_DIR / "servers.json"
    if not servers_path.exists():
        raise FileNotFoundError(f"Servers config not found: {servers_path}\nRun step1 first.")

    servers = load_json(servers_path)
    print(f"  ✓ Loaded {len(servers)} servers")

    # Get all available tools
    tools_by_server = get_all_tools_from_servers(servers)
    total_tools = sum(len(tools) for tools in tools_by_server.values())
    print(f"  ✓ Found {total_tools} tools across all servers")

    # Validate tasks
    print("\n🔍 Validating tasks...")
    valid_tasks = []
    invalid_tasks = []

    for task in working_tasks:
        task_id = task.get("task_id", "unknown")[:8]
        issues = validate_task(task, tools_by_server)

        if issues:
            invalid_tasks.append((task_id, issues))
            print(f"    ⚠ {task_id}: {issues}")
        else:
            valid_tasks.append(task)
            print(f"    ✓ {task_id}: {task.get('required_servers')} → {task.get('required_tools')}")

    # Enrich tasks with metadata
    print("\n📝 Enriching tasks with metadata...")
    enriched_tasks = []
    for task in valid_tasks:
        enriched = enrich_task_with_metadata(task, DATA_DIR)
        enriched_tasks.append(enriched)

    # Create final task dataset
    print("\n💾 Creating final task dataset...")
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Save as the main benchmark dataset
    output_path = TASKS_DIR / "progressivemcpbench.json"
    save_json(output_path, enriched_tasks)

    # Also create a summary file
    summary = {
        "total_tasks": len(enriched_tasks),
        "servers_used": list(set(s for t in enriched_tasks for s in t.get("required_servers", []))),
        "tools_used": list(set(tool for t in enriched_tasks for tool in t.get("required_tools", []))),
        "categories": list(set(t.get("category") for t in enriched_tasks if t.get("category"))),
        "tasks_by_category": {},
    }

    for task in enriched_tasks:
        cat = task.get("category", "Unknown")
        summary["tasks_by_category"][cat] = summary["tasks_by_category"].get(cat, 0) + 1

    save_json(TASKS_DIR / "summary.json", summary)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Valid tasks: {len(valid_tasks)}")
    print(f"  Invalid tasks: {len(invalid_tasks)}")
    print(f"  Servers used: {len(summary['servers_used'])}")
    print(f"  Tools used: {len(summary['tools_used'])}")
    print(f"\n  Categories:")
    for cat, count in summary["tasks_by_category"].items():
        print(f"    {cat}: {count}")

    if invalid_tasks:
        print(f"\n  ⚠ {len(invalid_tasks)} tasks have issues:")
        for task_id, issues in invalid_tasks:
            print(f"    {task_id}: {issues}")

    print(f"\n✅ Step 3 complete!")
    print(f"   Output: {output_path}")
    print(f"   Next: Run step4_validate_all.py")


if __name__ == "__main__":
    main()
