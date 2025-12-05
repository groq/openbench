"""
MCP Server Manager for ProgressiveMCPBench.

Handles starting and stopping the synthetic MCP HTTP server automatically
when running evaluations.
"""

import atexit
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

_server_process: subprocess.Popen | None = None
_server_lock = Lock()

SYNTHETIC_MCP_DIR = Path(__file__).parent.parent.parent.parent / "synthetic_mcp"
SERVER_SCRIPT = SYNTHETIC_MCP_DIR / "server" / "http_mcp_server.py"
DEFAULT_PORT = 8765
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
    """Wait for the server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port, host):
            return True
        time.sleep(0.1)
    return False


def start_mcp_server(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> bool:
    """Start the synthetic MCP HTTP server if not already running.

    Returns True if the server is running (either started or was already running).
    """
    global _server_process

    with _server_lock:
        if is_port_in_use(port, host):
            return True

        if not SERVER_SCRIPT.exists():
            raise FileNotFoundError(
                f"MCP server script not found at {SERVER_SCRIPT}. "
                "Make sure the synthetic_mcp directory is present."
            )

        _server_process = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(SYNTHETIC_MCP_DIR),
        )

        atexit.register(stop_mcp_server)

        if not wait_for_server(port, host):
            stop_mcp_server()
            raise RuntimeError(
                f"MCP server failed to start on {host}:{port} within timeout"
            )

        return True


def stop_mcp_server() -> None:
    """Stop the synthetic MCP HTTP server if running."""
    global _server_process

    with _server_lock:
        if _server_process is not None:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait()
            _server_process = None


def ensure_mcp_server_running(
    port: int = DEFAULT_PORT, host: str = DEFAULT_HOST
) -> None:
    """Ensure the MCP server is running, starting it if necessary.

    This is the main entry point for evaluations to use.
    """
    start_mcp_server(port, host)
