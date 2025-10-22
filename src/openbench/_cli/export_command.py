from typing import Annotated, Optional, List, Dict, Any
import os
import typer
from openbench._cli.export import (
    _read_log_json,
    _flatten_results,
    _flatten_stats,
    _flatten_samples,
    _generate_config_prefix,
)
from datasets import Dataset  # type: ignore[import-untyped]


def run_export(
    log_files: Annotated[
        List[str],
        typer.Option(
            help="Log file(s) to export to HuggingFace Hub. Can specify multiple files.",
        ),
    ],
    hub_repo: Annotated[
        str,
        typer.Option(
            help="Target Hub dataset repo (e.g. username/openbench-logs)",
            envvar="BENCH_HUB_REPO",
        ),
    ],
    hub_private: Annotated[
        bool,
        typer.Option(
            help="Create/update the Hub dataset as private",
            envvar="BENCH_HUB_PRIVATE",
        ),
    ] = False,
    hub_benchmark_name: Annotated[
        Optional[str],
        typer.Option(
            help="Override benchmark name in Hub config (default: auto-detect from task)",
            envvar="BENCH_HUB_BENCHMARK_NAME",
        ),
    ] = None,
    hub_model_name: Annotated[
        Optional[str],
        typer.Option(
            help="Override model name in Hub config (default: auto-detect from model)",
            envvar="BENCH_HUB_MODEL_NAME",
        ),
    ] = None,
) -> None:
    """Export evaluation log files to HuggingFace Hub.

    Example:
        bench export-hf --log-files logs/my_eval.eval --hub-repo username/openbench-logs
        bench export-hf --log-files logs/file1.eval --log-files logs/file2.eval --hub-repo username/openbench-logs --hub-private
    """
    if not log_files:
        typer.secho("No log files specified", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Validate log files exist
    valid_files = []
    for log_file in log_files:
        if os.path.exists(log_file):
            valid_files.append(os.path.abspath(log_file))
        else:
            typer.secho(
                f"Warning: Log file not found: {log_file}", fg=typer.colors.YELLOW
            )

    if not valid_files:
        typer.secho("No valid log files found to export", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Exporting {len(valid_files)} eval log(s) to {hub_repo}")

    results_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []
    samples_rows: List[Dict[str, Any]] = []
    first_valid_log_data: Dict[str, Any] = {}

    for path in valid_files:
        try:
            data = _read_log_json(path)
        except Exception as e:
            typer.secho(f"Skipping log '{path}': {e}", fg=typer.colors.YELLOW)
            continue

        if not first_valid_log_data:
            first_valid_log_data = data

        eval_info = data.get("eval", {})
        base = {
            "log_path": path,
            "eval_id": eval_info.get("eval_id"),
            "run_id": eval_info.get("run_id"),
            "created": eval_info.get("created"),
            "task": eval_info.get("task"),
            "task_id": eval_info.get("task_id"),
            "model": eval_info.get("model"),
        }

        results_rows.append(_flatten_results(data, base))
        stats_rows.extend(_flatten_stats(data, base))
        samples_rows.extend(_flatten_samples(data, base))

    if not first_valid_log_data:
        typer.secho("No valid logs could be parsed", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Generate config name prefix for this run
    config_prefix = _generate_config_prefix(
        hub_benchmark_name, hub_model_name, first_valid_log_data
    )

    if results_rows:
        ds = Dataset.from_list(results_rows)
        config_name = f"{config_prefix}_results"
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name=config_name,
            split="train",
            private=hub_private,
        )
        typer.echo(
            f"Pushed results ({len(results_rows)} rows) to {hub_repo} [config={config_name}]"
        )

    if stats_rows:
        ds = Dataset.from_list(stats_rows)
        config_name = f"{config_prefix}_stats"
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name=config_name,
            split="train",
            private=hub_private,
        )
        typer.echo(
            f"Pushed stats ({len(stats_rows)} rows) to {hub_repo} [config={config_name}]"
        )

    if samples_rows:
        ds = Dataset.from_list(samples_rows)
        config_name = f"{config_prefix}_samples"
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name=config_name,
            split="train",
            private=hub_private,
        )
        typer.echo(
            f"Pushed samples ({len(samples_rows)} rows) to {hub_repo} [config={config_name}]"
        )

    typer.secho("✅ Export complete!", fg=typer.colors.GREEN)
