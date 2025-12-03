# Mock MCP Layer Design

## Overview

This document describes a mock MCP layer that replaces all real MCP servers with a deterministic, log-backed simulator. The goal is to eliminate infrastructure failures (LibreOffice not installed, DuckDuckGo rate limiting, chart generation failing, HTTP 403s, etc.) while preserving the evaluation's core question: **"Did the model pick and use the right tools?"**

## Design Principles

### 1. No Meta-Tool Mocking

Meta-tools (`route`, `execute-tool`, `ls`, `read-tool-file`) are executed instantly in Python coordination code. They don't dispatch to real MCP servers and don't need mocking. The copilot router's semantic search remains as-is—a bad query to the router is just as wrong as a bad tool call.

### 2. Completely Stateless

The mock layer is entirely stateless:
- No tracking of state changes between tool calls
- Filesystem listings return frozen-in-time snapshots
- Mutation tools (write_file, create_pdf, etc.) return success responses but do nothing
- Each tool call is answered independently based only on its arguments

### 3. Strategy-Independent

The mock layer's behavior is identical regardless of strategy (copilot, directory, minimal-tools, etc.). Strategy only influences which tools are presented to the model. The MCP simulation layer knows nothing about strategies.

### 4. Task-Independent

The mock layer is not aware of `task_id`. We're building a realistic simulation of MCP servers, not a precise replay of previous runs. This allows:
- Tweaking task prompts without breaking the mock layer
- Adding new tasks that use existing tools
- Iterating on the benchmark without re-collecting data

### 5. Required Tools vs Expected Tools

These are separate concepts:
- **required_tools**: Hand-curated annotation in task metadata. Input to minimal-tools strategy. Not derived from logs.
- **expected_tools**: Union of all tools successfully used across runs. Used by mock layer to distinguish expected-path from distraction calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Inspect AI Eval                         │
├─────────────────────────────────────────────────────────────┤
│  Strategy Layer (unchanged)                                 │
│  ├─ copilot_tool_source() ─────► route(), execute-tool()   │
│  ├─ directory_tool_source() ──► ls(), read-tool-file()     │
│  ├─ minimal_tools_tool_source()                            │
│  └─ distraction_*_tool_source()                            │
├─────────────────────────────────────────────────────────────┤
│  Mock MCP Router (NEW)                                      │
│  ├─ Canonical Response Store (SQLite)                       │
│  ├─ Argument Canonicalizer                                  │
│  └─ LLM Fallback (gpt-oss-20b)                             │
└─────────────────────────────────────────────────────────────┘
```

## SQLite Schema

```sql
-- Canonical tool call/response pairs from successful runs
CREATE TABLE tool_responses (
    id INTEGER PRIMARY KEY,
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    canonical_args TEXT NOT NULL,  -- JSON, sorted keys, normalized paths
    response_text TEXT NOT NULL,   -- Raw response content
    response_size INTEGER,         -- For stats/filtering
    source_eval TEXT,              -- Which .eval file this came from
    source_sample TEXT,            -- Which sample ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(server_name, tool_name, canonical_args)
);

-- Index for fast lookups
CREATE INDEX idx_tool_lookup ON tool_responses(server_name, tool_name, canonical_args);

-- Set of tools that have been successfully used (for expected vs distraction)
CREATE TABLE expected_tools (
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    call_count INTEGER DEFAULT 1,  -- How often this tool was used successfully
    PRIMARY KEY (server_name, tool_name)
);

-- Example responses for distraction tool fabrication
CREATE TABLE distraction_examples (
    id INTEGER PRIMARY KEY,
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_description TEXT,
    example_args TEXT,            -- JSON
    example_response TEXT,        -- What a realistic response looks like
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool schemas for LLM prompting
CREATE TABLE tool_schemas (
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    description TEXT,
    input_schema TEXT,            -- JSON schema
    PRIMARY KEY (server_name, tool_name)
);
```

## Response Strategy

### Tier 1: Canonical Replay (No LLM)

For each tool call:
1. Canonicalize arguments (sort keys, normalize paths, strip whitespace)
2. Look up `(server_name, tool_name, canonical_args)` in SQLite
3. If found: return exact stored response

This handles ~90% of expected-path calls with zero latency and bit-identical responses.

```python
def mock_execute(server_name: str, tool_name: str, params: dict) -> str:
    canonical_args = canonicalize(server_name, tool_name, params)
    
    row = db.execute("""
        SELECT response_text FROM tool_responses
        WHERE server_name = ? AND tool_name = ? AND canonical_args = ?
    """, (server_name, tool_name, canonical_args)).fetchone()
    
    if row:
        return row["response_text"]
    
    # Fall through to Tier 2 or 3...
```

### Tier 2: Fuzzy Match for Expected Tools (LLM-Assisted)

If no exact match but tool is in `expected_tools`:
1. Retrieve all stored examples for this `(server_name, tool_name)`
2. Ask gpt-oss-20b which example semantically matches the new call
3. Return that example's response verbatim (no fabrication)

```python
def fuzzy_match_expected(server_name: str, tool_name: str, params: dict) -> str | None:
    # Get all examples for this tool
    examples = db.execute("""
        SELECT canonical_args, response_text FROM tool_responses
        WHERE server_name = ? AND tool_name = ?
        LIMIT 10
    """, (server_name, tool_name)).fetchall()
    
    if not examples:
        return None
    
    # Ask LLM to match
    prompt = build_fuzzy_match_prompt(server_name, tool_name, params, examples)
    result = llm_call(prompt)
    
    # LLM returns index of matching example, or "NONE"
    if result.isdigit() and int(result) < len(examples):
        return examples[int(result)]["response_text"]
    
    return None  # No match, will fall through to error response
```

**Fuzzy Match Prompt:**

```
You are matching a new tool call to stored examples.

Tool: {server_name}/{tool_name}

Stored examples:
[0] args: {"path": "/root/pdf/paper1.pdf"}
    response: "Title: Embodied AI Survey..."

[1] args: {"path": "/root/pdf/paper2.pdf"}  
    response: "Title: Robot Learning..."

New call args: {"path": "/root/pdf/paper1.pdf", "page": 1}

If the new call is semantically equivalent to one of the examples (same resource, 
compatible arguments), output ONLY the example number (e.g., "0").
If no example matches, output "NONE".

Answer:
```

### Tier 3: Distraction Tool Fabrication (LLM)

If tool is NOT in `expected_tools`:
1. Retrieve example responses for this tool type from `distraction_examples`
2. Ask gpt-oss-20b to generate a plausible response matching the shape
3. Never leak task-solving information

```python
def fabricate_distraction(server_name: str, tool_name: str, params: dict) -> str:
    # Get schema and examples
    schema = get_tool_schema(server_name, tool_name)
    examples = get_distraction_examples(server_name, tool_name)
    
    prompt = build_distraction_prompt(server_name, tool_name, schema, params, examples)
    return llm_call(prompt)
```

**Distraction Fabrication Prompt:**

```
You are simulating an MCP tool response. Generate a plausible response that:
- Matches the expected output shape for this tool type
- Contains realistic-looking but generic content
- Does NOT solve any specific user task
- Is concise (<200 tokens)

Tool: {server_name}/{tool_name}
Description: {description}
Input Schema: {input_schema}
Arguments: {params}

Example responses from similar tools:
---
{example_1}
---
{example_2}
---

Generate a response in the same style. Output ONLY the tool response, no explanation.
```

### Tier 4: Error Response for Unmatched Expected Tools

If an expected-path tool is called with completely novel arguments and fuzzy matching fails:
- Return a plausible error response ("file not found", "invalid parameters")
- This correctly fails the task, as it would have in reality

```python
def error_response(server_name: str, tool_name: str, params: dict) -> str:
    return json.dumps({
        "error": True,
        "message": f"Resource not found or invalid parameters for {tool_name}",
        "params": params
    })
```

## Data Ingestion Pipeline

### Phase 1: Extract Successful Tool Calls

```python
def ingest_eval_file(eval_path: Path, db: sqlite3.Connection):
    """Extract tool call/response pairs from successful samples."""
    
    with zipfile.ZipFile(eval_path) as zf:
        # Parse each sample
        for sample_file in zf.namelist():
            if not sample_file.startswith("samples/"):
                continue
            
            sample = json.load(zf.open(sample_file))
            
            # Only process successful samples
            score = sample.get("scores", {}).get("progressivemcpbench_scorer", {}).get("value", 0)
            if score == 0:
                continue
            
            # Walk message history
            messages = sample.get("messages", [])
            pending_calls = {}
            
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        call_id = tc.get("id")
                        pending_calls[call_id] = tc
                
                elif msg.get("role") == "tool":
                    call_id = msg.get("tool_call_id")
                    if call_id in pending_calls:
                        tc = pending_calls.pop(call_id)
                        
                        # Extract server/tool from the call
                        server_name, tool_name, params = parse_tool_call(tc)
                        response_text = extract_response_text(msg)
                        
                        if server_name and tool_name:
                            insert_response(db, server_name, tool_name, params, 
                                          response_text, eval_path.name, sample["id"])
```

### Phase 2: Build Expected Tools Set

```python
def build_expected_tools(db: sqlite3.Connection):
    """Derive expected_tools from all successfully used tools."""
    
    db.execute("""
        INSERT OR REPLACE INTO expected_tools (server_name, tool_name, call_count)
        SELECT server_name, tool_name, COUNT(*) 
        FROM tool_responses
        GROUP BY server_name, tool_name
    """)
```

### Phase 3: Collect Distraction Examples

```python
def collect_distraction_examples(db: sqlite3.Connection):
    """
    Collect example responses for tools that might be used as distractors.
    These come from:
    1. Successful calls in non-minimal strategies (distraction-64, distraction-128)
    2. Failed calls where the tool itself worked but wasn't the right choice
    """
    
    # Also collect tool schemas from the MCP server definitions
    for server in ALL_MCP_SERVERS:
        for tool in server.tools:
            db.execute("""
                INSERT OR REPLACE INTO tool_schemas 
                (server_name, tool_name, description, input_schema)
                VALUES (?, ?, ?, ?)
            """, (server.name, tool.name, tool.description, json.dumps(tool.input_schema)))
```

## Argument Canonicalization

```python
def canonicalize(server_name: str, tool_name: str, params: dict) -> str:
    """
    Normalize arguments to a canonical form for lookup.
    """
    normalized = {}
    
    for key, value in params.items():
        # Normalize paths
        if "path" in key.lower() or "file" in key.lower():
            if isinstance(value, str):
                value = os.path.normpath(value)
                value = value.rstrip("/")
        
        # Lowercase certain string fields
        if key.lower() in ("server_name", "tool_name"):
            value = value.lower() if isinstance(value, str) else value
        
        normalized[key] = value
    
    # Sort keys for consistent JSON
    return json.dumps(normalized, sort_keys=True)
```

### Per-Tool Canonicalization Overrides

Some tools need special handling:

```python
CANONICALIZERS = {
    ("filesystem", "read_file"): lambda p: {
        "path": os.path.normpath(p.get("path", p.get("file_path", "")))
    },
    ("excel", "excel_read_sheet"): lambda p: {
        "path": os.path.normpath(p.get("fileAbsolutePath", p.get("path", ""))),
        "sheet": p.get("sheetName", p.get("sheet", "Sheet1"))
    },
    # Add more as needed...
}
```

## Mutation Tool Handling

Tools that claim to mutate state return success but do nothing:

```python
MUTATION_TOOLS = {
    ("filesystem", "write_file"),
    ("filesystem", "create_directory"),
    ("pdf-generator", "create_pdf"),
    ("mcp-server-chart", "generate_bar_chart"),
    ("mcp-server-chart", "generate_pie_chart"),
    # ...
}

def handle_mutation_tool(server_name: str, tool_name: str, params: dict) -> str:
    """Return a plausible success response without actually doing anything."""
    
    if (server_name, tool_name) == ("filesystem", "write_file"):
        path = params.get("path", "/tmp/mock_file.txt")
        return json.dumps({"success": True, "path": path, "bytes_written": len(params.get("content", ""))})
    
    if "chart" in tool_name.lower():
        return json.dumps({"success": True, "path": f"/tmp/mock_{tool_name}_{hash(str(params)) % 10000}.png"})
    
    # Generic success
    return json.dumps({"success": True})
```

## Implementation Plan

### Step 1: Create Ingestion Script
- `scripts/ingest_mock_data.py`
- Parses all successful samples from .eval files
- Populates SQLite database with tool_responses, expected_tools, tool_schemas
- Run once to build the initial database

### Step 2: Create Mock MCP Router
- `src/openbench/evals/progressivemcpbench_mock.py` or similar
- Implements `mock_execute()` with the 4-tier strategy
- Handles argument canonicalization
- Manages LLM calls for fuzzy matching and distraction fabrication

### Step 3: Create Mock Tool Sources
- Modify existing `*_tool_source()` functions to optionally use mock router
- Add `--mock` flag to eval command
- Or create parallel `mock_*_tool_source()` functions

### Step 4: Collect Distraction Examples
- Separate pass over logs to collect diverse tool response examples
- Include examples from failed runs where tools returned valid data
- Store in distraction_examples table

### Step 5: Testing & Validation
- Run eval with mock layer on a subset of tasks
- Compare scores to real MCP runs
- Tune canonicalizers based on cache miss patterns
- Adjust LLM prompts as needed

## Expected Outcomes

1. **Elimination of infrastructure failures** (~15% of current failures)
2. **Faster eval runs** (no network calls, no slow MCP servers)
3. **Deterministic results** (same inputs → same outputs)
4. **Easier iteration** (can modify tasks without breaking mock layer)
5. **Lower cost** (only LLM calls for cache misses and distractions)

## Open Questions

1. **How to handle the scorer?** If scorer expects real files to exist, we may need to materialize mutation results to disk, or modify scorer to accept mock responses.

2. **Cross-task response sharing?** If `/root/pdf/paper1.pdf` is read in multiple tasks, should we share the response? Current design does this implicitly through `(server, tool, canonical_args)` keying.

3. **LLM temperature for distractions?** Should be 0 for reproducibility, but might produce repetitive responses. Consider temperature 0.3 with seed.

4. **Handling new tools?** If a new MCP server is added to the benchmark, it will initially have no examples. Need a fallback to real MCP or manual example collection.
