from __future__ import annotations
from pathlib import Path
from typing import Any
import os
import json
import sqlite3
import logging

import mcp.types as types
from mcp.server.fastmcp import FastMCP, Context

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# Env vars
_DB_PATH = Path(os.getenv("PROGRESSIVE_MCP_DB", "progressive_mcpbench.sqlite"))
_STRATEGY = os.getenv("PROGRESSIVE_MCP_STRATEGY", "all")  # 'all', 'all-relevant', 'minimal'
_RELEVANT_SERVER = os.getenv("PROGRESSIVE_MCP_RELEVANT_SERVER")

server = FastMCP("mock-mcp")

def _open_conn() -> sqlite3.Connection:
    if not _DB_PATH.exists():
        raise RuntimeError(f"Database not found at {_DB_PATH}")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _get_used_tools(session_id: int) -> set[tuple[str, str]]:
    conn = _open_conn()
    try:
        cur = conn.execute(
            """
            SELECT request_json FROM tool_calls
            WHERE session_id = ? AND call_type = 'execute'
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        used = set()
        for row in rows:
            req = json.loads(row["request_json"])
            used.add((req["server_name"], req["tool_name"]))
        return used
    finally:
        conn.close()

def _find_matching_route_calls(query: str) -> list[sqlite3.Row]:
    conn = _open_conn()
    try:
        # Strict match on query text
        cur = conn.execute(
            """
            SELECT * FROM tool_calls
            WHERE call_type = 'route' AND json_extract(request_json, '$.query') = ?
            ORDER BY id ASC
            """,
            (query,),
        )
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()

def _find_matching_execute_call(server_name: str, tool_name: str, params: dict[str, Any]) -> sqlite3.Row | None:
    conn = _open_conn()
    try:
        # We fetch all executions for this server/tool and find the matching params in Python
        # This avoids JSON formatting issues in SQL
        cur = conn.execute(
            """
            SELECT * FROM tool_calls
            WHERE call_type = 'execute'
              AND json_extract(request_json, '$.server_name') = ?
              AND json_extract(request_json, '$.tool_name') = ?
            ORDER BY id ASC
            """,
            (server_name, tool_name),
        )
        rows = cur.fetchall()
        
        for row in rows:
            req = json.loads(row["request_json"])
            # Compare params dictionaries
            if req.get("params") == params:
                return row
        
        return None
    finally:
        conn.close()

@server.tool(
    name="route",
    description=(
        """
This is a tool used to find MCP servers and tools that can solve user needs    
    When to use this tool:
        -When faced with user needs, you (LLM) are unable to solve them on your own and do not have the tools to solve the problem.
        -When a user proposes a new task and you (LLM) are unsure which specific tool to use to complete it.
        -When the user's request is vague or complex, and feasible tool options need to be explored first.
        -This is the first step in executing unknown tasks, known as the "discovery" phase, aimed at finding the correct tool.
    **Parameter Description**
    Query (string, required): The input query must contain a <tool_assistant> tag with server and tool descriptions, for example: 
        <tool_assistant>
        server: ... # Platform/permission domain
        tool: ... # Operation type + target
        </tool_assistant>
"""
    ),
)
async def route(query: str, ctx: Context | None = None) -> types.CallToolResult:
    rows = _find_matching_route_calls(query)
    if not rows:
        logger.warning(f"Cache miss for route query: {query}")
        return types.CallToolResult(
            isError=True,
            content=[types.TextContent(type="text", text="No recorded route for this query (MockMCP cache miss)")],
        )

    # Strategy handling
    # For now, we just return the first match regardless of strategy
    # In "all" strategy, we might want to combine results if multiple routes were recorded?
    # But usually route is called once per task step.
    # If multiple sessions have same query, we pick the first one.
    
    row = rows[0]
    result = json.loads(row["result_json"])
    
    # Parse content to filter tools
    content_list = result.get("content", [])
    if not content_list or content_list[0].get("type") != "text":
         return types.CallToolResult.model_validate(result)
         
    yaml_text = content_list[0]["text"]
    
    data = None
    if yaml:
        try:
            data = yaml.safe_load(yaml_text)
        except:
            pass
            
    if data is None:
        try:
            data = json.loads(yaml_text)
        except:
            # Can't parse, return as is
            return types.CallToolResult.model_validate(result)
            
    if not data or "matched_tools" not in data:
         return types.CallToolResult.model_validate(result)
         
    matched_tools = data["matched_tools"]
    filtered_tools = []
    
    if _STRATEGY == "minimal":
        session_id = row["session_id"]
        used_tools = _get_used_tools(session_id)
        for tool in matched_tools:
             if (tool.get("server_name"), tool.get("tool_name")) in used_tools:
                 filtered_tools.append(tool)
                 
    elif _STRATEGY == "all-relevant":
        if _RELEVANT_SERVER:
            for tool in matched_tools:
                if tool.get("server_name") == _RELEVANT_SERVER:
                    filtered_tools.append(tool)
        else:
            filtered_tools = matched_tools
            
    else: # "all"
        filtered_tools = matched_tools
        
    data["matched_tools"] = filtered_tools
    
    # Dump back
    if yaml:
        new_yaml = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        new_yaml = json.dumps(data, indent=2)
        
    result["content"][0]["text"] = new_yaml
    
    # Validate types
    return types.CallToolResult.model_validate(result)


@server.tool(
    name="execute-tool",
    description=(
        """
A tool for executing a specific tool on a specific server.Select tools only from the results obtained from the previous route each time.

When to use this tool:
    - When using the route tool to route to a specific MCP server and tool
    - When the 'execute-tool' fails to execute (up to 3 repetitions).
    - When the user's needs and previous needs require the same tool.

Parameters explained:
    -server_name: string, required. The name of the server where the target tool is located.

    -tool_name: string, required. The name of the target tool to be executed.

    -params: dictionary or None, optional. A dictionary containing all parameters that need to be passed to the target tool. This can be omitted if the target tool does not require parameters.
"""
    ),
)
async def execute_tool(
    server_name: str,
    tool_name: str,
    params: dict[str, Any] | None,
    ctx: Context | None = None,
) -> types.CallToolResult:
    
    row = _find_matching_execute_call(server_name, tool_name, params or {})
    if not row:
        logger.warning(f"Cache miss for execute-tool: {server_name}.{tool_name} params={params}")
        return types.CallToolResult(
            isError=True,
            content=[types.TextContent(
                type="text",
                text=f"No recorded execution for {server_name}.{tool_name} with given params (MockMCP cache miss)",
            )],
        )
    
    result = json.loads(row["result_json"])
    return types.CallToolResult.model_validate(result)

# Patch for "all" strategy to load tools dynamically
if _STRATEGY == "all":
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, ArgModelBase
    from typing import Callable
    
    class PassthroughFuncMetadata(FuncMetadata):
        async def call_fn_with_arg_validation(
            self,
            fn: Callable,
            fn_is_async: bool,
            arguments_to_validate: dict[str, Any],
            arguments_to_pass_directly: dict[str, Any] | None,
        ) -> Any:
            kwargs = arguments_to_validate.copy()
            if arguments_to_pass_directly:
                kwargs.update(arguments_to_pass_directly)
            
            if fn_is_async:
                return await fn(**kwargs)
            else:
                return fn(**kwargs)

    TOOLS_JSON_PATH = Path(os.path.expanduser("~/.openbench/progressive_mcp_bench/config/tools.json"))
    
    if TOOLS_JSON_PATH.exists():
        try:
            with open(TOOLS_JSON_PATH) as f:
                tools_data = json.load(f)
            
            for item in tools_data:
                if "tools" in item:
                    for srv_name, srv_tools in item["tools"].items():
                        for tool_def in srv_tools.get("tools", []):
                            tool_name = tool_def["name"]
                            full_name = f"{srv_name}.{tool_name}"
                            
                            # Correct way to make closure in loop
                            def create_handler(s, t):
                                async def handler(**kwargs):
                                    row = _find_matching_execute_call(s, t, kwargs)
                                    if not row:
                                        # Try to return a helpful error message
                                        return f"Error: No recorded execution found for {s}.{t} with these parameters."
                                    
                                    result_json = json.loads(row["result_json"])
                                    content = result_json.get("content", [])
                                    
                                    # Extract text if possible
                                    texts = [c["text"] for c in content if c["type"] == "text"]
                                    return "\n".join(texts)
                                return handler
                            
                            handler = create_handler(srv_name, tool_name)
                            handler.__name__ = full_name.replace(".", "_")
                            
                            # Create metadata
                            meta = PassthroughFuncMetadata(
                                arg_model=ArgModelBase # Dummy
                            )
                            
                            tool = Tool(
                                fn=handler,
                                name=full_name,
                                description=tool_def.get("description", "") or "",
                                parameters=tool_def.get("inputSchema", {}),
                                fn_metadata=meta,
                                is_async=True
                            )
                            
                            server._tool_manager._tools[tool.name] = tool

        except Exception as e:
            logger.error(f"Failed to load tools from {TOOLS_JSON_PATH}: {e}")

if __name__ == "__main__":
    server.run(transport="stdio")
