"""
MCP Server Manager for ProgressiveMCPBench.

This module verifies connectivity to the synthetic MCP server.
By default, it connects to the production remote server; use environment
variables to override for local development.
"""

import json
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse

from openbench.tools.progressivemcpbench.mcp_config import get_mcp_base_url


def _test_server_connectivity(base_url: str, timeout: float = 5.0) -> bool:
    """Test if the MCP server is responding.

    Args:
        base_url: Base URL for the MCP server
        timeout: Connection timeout in seconds

    Returns:
        True if server responds successfully, False otherwise
    """
    parsed = urlparse(base_url)
    use_https = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if use_https else 80)

    conn: HTTPConnection | HTTPSConnection
    if use_https:
        conn = HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = HTTPConnection(host, port, timeout=timeout)

    try:
        test_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "connectivity-test", "version": "1.0.0"},
            },
        }
        body = json.dumps(test_data)
        conn.request(
            "POST",
            "/mcp/filesystem",
            body,
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        response = conn.getresponse()
        return response.status == 200
    except Exception:
        return False
    finally:
        conn.close()


def ensure_mcp_server_running(
    base_url: str | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> None:
    """Ensure the MCP server is accessible.

    Uses the centralized configuration by default, connecting to the production
    remote server. Use PROGRESSIVE_MCP_LOCAL=1 or PROGRESSIVE_MCP_URL to override.

    Args:
        base_url: Base URL for the MCP server (defaults to configured URL)
        retries: Number of connection attempts
        retry_delay: Delay between retries in seconds

    Raises:
        RuntimeError: If the server is not accessible after all retries
    """
    resolved_url = base_url if base_url is not None else get_mcp_base_url()

    for attempt in range(retries):
        if _test_server_connectivity(resolved_url):
            return

        if attempt < retries - 1:
            time.sleep(retry_delay)

    raise RuntimeError(
        f"MCP server is not accessible at {resolved_url}. "
        "For local development, ensure the server is running and set "
        "PROGRESSIVE_MCP_LOCAL=1. Otherwise, check your network connection."
    )
