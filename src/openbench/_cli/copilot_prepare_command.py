from pathlib import Path
import typer

from openbench.tools.livemcpbench.copilot.prepare import (
    prepare_copilot_cache,
    prepare_root_data,
    install_playwright_browsers,
)


def run_copilot_prepare(
    refresh: bool = typer.Option(
        False,
        "--refresh",
        "-r",
        help="Force re-fetch upstream JSONs and re-generate embeddings",
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Optional output path for the embeddings JSON"
    ),
    root_data: bool = typer.Option(
        False,
        "--root-data",
        help="Also fetch and stage annotated_data into the root sandbox",
    ),
    playwright_install: bool = typer.Option(
        False, "--playwright-install", help="Install Playwright browsers via npx"
    ),
    browsers: str = typer.Option(
        "chromium",
        "--browsers",
        help="Comma-separated list of Playwright browsers to install (chromium,firefox,webkit)",
    ),
    browsers_path: str | None = typer.Option(
        None,
        "--browsers-path",
        help="Optional path for PLAYWRIGHT_BROWSERS_PATH (defaults to ~/.cache/openbench/playwright-browsers)",
    ),
) -> None:
    """Prepare caches for the MCP Copilot (upstream JSONs + embeddings)."""
    out_path = Path(output) if output else None
    try:
        path = prepare_copilot_cache(force_refresh=refresh, embeddings_path=out_path)
        typer.echo(f"✅ Copilot cache ready: {path}")
        if root_data:
            root_path = prepare_root_data(force_refresh=refresh)
            typer.echo(f"✅ Root sandbox ready: {root_path}")
        if playwright_install:
            try:
                blist = [b.strip() for b in browsers.split(",") if b.strip()]
                bp = install_playwright_browsers(
                    browsers=blist,
                    browsers_path=Path(browsers_path) if browsers_path else None,
                )
                typer.echo(f"✅ Playwright browsers installed at: {bp}")
            except Exception as e:
                typer.echo(f"❌ Failed to install Playwright browsers: {e}")
                raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to prepare Copilot cache: {e}")
        raise typer.Exit(1)
