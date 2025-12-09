"""
MCP Server Manager for ProgressiveMCPBench.

This module manages connectivity to the external synthetic MCP HTTP server
that runs on port 9123 (in progressivemcpbench project), instead of starting
a local Python server.

The server should be started externally and this module will verify connectivity.
"""

import socket
import time
import json
from pathlib import Path
from typing import Any

# Server is now externally managed
DEFAULT_PORT = 9123
DEFAULT_HOST = "localhost"


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def wait_for_server(port: int, host: str = "localhost", timeout: float = 10.0) -> bool:
    """Wait for the server to become available and test MCP connectivity."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_in_use(port, host):
            time.sleep(0.1)
            continue

        # Test actual MCP server response
        try:
            import http.client
            conn = http.client.HTTPConnection(host, port, timeout=2)
            try:
                # Test MCP initialization endpoint
                test_data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "connectivity-test", "version": "1.0.0"}
                    }
                }
                body = json.dumps(test_data)
                conn.request(
                    "POST",
                    "/mcp/filesystem",
                    body,
                    {
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    }
                )
                response = conn.getresponse()
                if response.status == 200:
                    return True
            except Exception:
                pass
            finally:
                conn.close()
        except Exception:
            pass

        time.sleep(0.1)
    return False


def ensure_mcp_server_running(
    port: int = DEFAULT_PORT, host: str = DEFAULT_HOST
) -> None:
    """Ensure the MCP server is running, checking connectivity to external server.

    This now verifies that the external MCP server is accessible rather than
    starting a local Python server process.

    Args:
        port: Port to check for MCP server
        host: Host to check for MCP server

    Raises:
        RuntimeError: If the server is not accessible
    """
    if not is_port_in_use(port, host):
        raise RuntimeError(
            f"MCP server is not accessible at {host}:{port}. "
            "Please ensure the external MCP server is running."
        )

    # Test actual MCP server response
    if not wait_for_server(port, host):
        raise RuntimeError(
            f"MCP server is not responding at {host}:{port}. "
            "Please check that the server is running and accessible."
        )