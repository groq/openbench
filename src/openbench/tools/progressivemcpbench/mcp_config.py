"""
Configuration for the ProgressiveMCPBench synthetic MCP server.

Provides centralized URL configuration with environment variable overrides
for local development.
"""

import os

# Production URL for the synthetic MCP server
PRODUCTION_MCP_URL = "https://progressive-mcp-bench.groq-dev.workers.dev"

# Local development URL
LOCAL_MCP_URL = "http://localhost:9123"


def get_mcp_base_url() -> str:
    """Get the base URL for the synthetic MCP server.

    Environment variables (in order of precedence):
        - PROGRESSIVE_MCP_URL: Explicit full URL override
        - PROGRESSIVE_MCP_LOCAL: If "1" or "true", use localhost:9123

    Returns:
        The base URL (without trailing slash)
    """
    # Explicit URL override takes highest priority
    explicit_url = os.getenv("PROGRESSIVE_MCP_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    # Check for local development flag
    use_local = os.getenv("PROGRESSIVE_MCP_LOCAL", "").lower() in ("1", "true")
    if use_local:
        return LOCAL_MCP_URL

    # Default to production
    return PRODUCTION_MCP_URL


def get_mcp_host_port() -> tuple[str, int]:
    """Get the host and port for the synthetic MCP server.

    This is used by modules that need separate host/port values
    rather than a full URL.

    Returns:
        Tuple of (host, port)
    """
    url = get_mcp_base_url()

    # Parse the URL to extract host and port
    if url.startswith("https://"):
        host = url[8:]
        port = 443
    elif url.startswith("http://"):
        host = url[7:]
        port = 80
    else:
        host = url
        port = 80

    # Check for explicit port in host
    if ":" in host:
        parts = host.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1])

    return host, port


def is_using_https() -> bool:
    """Check if the MCP server URL uses HTTPS."""
    return get_mcp_base_url().startswith("https://")
