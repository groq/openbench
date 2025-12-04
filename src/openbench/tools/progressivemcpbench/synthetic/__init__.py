"""
Synthetic MCP strategy implementations for ProgressiveMCPBench.

These strategies route tool calls to a synthetic HTTP MCP server instead of
real MCP servers, providing deterministic and fast evaluation.

The HTTP server runs on localhost:8765 and masquerades as multiple MCP servers
via /mcp/{server_name}/tools/{tool_name} paths.
"""
