#!/usr/bin/env python3
"""
Step 5: Apply Stub Call Annotations

This script reads the annotated stub_calls.json file and applies the
handler_override specifications to servers.json.

Workflow:
1. Run an eval with minimal-servers strategy
2. Review synthetic_mcp/logs/stub_calls.json
3. For each entry, add one of:
   - "annotation": "Description of what this tool should return"
   - "handler_override": {"type": "static_json", "response": {...}}
   - "handler_override": {"type": "filesystem", "root": "/root"}
   - etc.
4. Run this script to apply overrides to servers.json

Handler types:
- static_json: Return a fixed JSON response
- filesystem: Read from synthetic filesystem
- table_lookup: Look up from JSON data file
- excel_reader: Read Excel files
- web_corpus: Fetch from synthetic web corpus
- url_search: Search URL index
"""

import json
from pathlib import Path
from typing import Any

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_MCP_DIR = SCRIPT_DIR.parent
LOGS_DIR = SYNTHETIC_MCP_DIR / "logs"
CONFIG_DIR = SYNTHETIC_MCP_DIR / "config"
STUB_CALLS_PATH = LOGS_DIR / "stub_calls.json"
SERVERS_PATH = CONFIG_DIR / "servers.json"


def load_json(path: Path) -> Any:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON to file with pretty printing."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {path}")


def apply_handler_override(
    servers: dict,
    server_name: str,
    tool_name: str,
    handler: dict,
) -> bool:
    """Apply a handler override to a tool in servers.json.
    
    Returns True if the override was applied successfully.
    """
    if server_name not in servers:
        print(f"  ⚠ Server '{server_name}' not found in servers.json")
        return False
    
    server = servers[server_name]
    for tool in server.get("tools", []):
        if tool.get("name") == tool_name:
            tool["handler"] = handler
            return True
    
    print(f"  ⚠ Tool '{tool_name}' not found in server '{server_name}'")
    return False


def main():
    print("=" * 60)
    print("Step 5: Apply Stub Call Annotations")
    print("=" * 60)
    
    # Check for stub calls file
    if not STUB_CALLS_PATH.exists():
        print(f"\n❌ No stub calls log found at {STUB_CALLS_PATH}")
        print("   Run an eval first to generate stub call data.")
        return
    
    # Load files
    print("\n📂 Loading files...")
    stub_calls_data = load_json(STUB_CALLS_PATH)
    servers = load_json(SERVERS_PATH)
    print(f"  ✓ Loaded {len(stub_calls_data.get('stub_calls', []))} stub call entries")
    print(f"  ✓ Loaded {len(servers)} servers")
    
    # Process entries with handler_override
    print("\n🔧 Processing annotations...")
    stub_calls = stub_calls_data.get("stub_calls", [])
    
    applied = 0
    annotated_only = 0
    skipped = 0
    
    for entry in stub_calls:
        server = entry.get("server", "")
        tool = entry.get("tool", "")
        handler_override = entry.get("handler_override")
        annotation = entry.get("annotation", "")
        
        if handler_override:
            # Apply the override
            if apply_handler_override(servers, server, tool, handler_override):
                print(f"  ✓ Applied override: {server}/{tool}")
                applied += 1
            else:
                skipped += 1
        elif annotation:
            # Has annotation but no handler yet
            print(f"  📝 Annotated but no handler: {server}/{tool}")
            print(f"     {annotation}")
            annotated_only += 1
        else:
            skipped += 1
    
    # Save updated servers.json
    if applied > 0:
        print("\n💾 Saving updated servers.json...")
        save_json(SERVERS_PATH, servers)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Handler overrides applied: {applied}")
    print(f"  Annotated (pending handler): {annotated_only}")
    print(f"  Skipped (no annotation): {skipped}")
    
    if annotated_only > 0:
        print("\n⚠ Some entries have annotations but no handler_override.")
        print("  Add handler_override to these entries and re-run this script.")
    
    print("\n✅ Step 5 complete!")


if __name__ == "__main__":
    main()
