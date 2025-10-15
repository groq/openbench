import subprocess
import sys
from typing import Optional
from pathlib import Path
import typer


def run_view(
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="Log directory to view (defaults to ./logs)",
        envvar="INSPECT_LOG_DIR",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Include all logs in log_dir recursively",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="TCP/IP host for server",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        help="TCP/IP port for server",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Set the log level",
    ),
) -> None:
    """
    View evaluation logs using inspect view.

    This is a wrapper around 'inspect view' that provides access to the log viewer.
    """
    cmd = ["inspect", "view"]

    # Add arguments if provided
    if log_dir:
        cmd.extend(["--log-dir", str(log_dir)])
    if not recursive:
        cmd.append("--no-recursive")
    if host:
        cmd.extend(["--host", host])
    if port:
        cmd.extend(["--port", str(port)])
    if log_level:
        cmd.extend(["--log-level", log_level])

    # Run the command
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Error running inspect view: {e}", err=True)
        sys.exit(1)
