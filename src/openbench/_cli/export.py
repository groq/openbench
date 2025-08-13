from __future__ import annotations

from typing import Optional, List, Dict, Any
import os
import json
import typer
from datasets import Dataset  # type: ignore[import-untyped]


def _read_log_json(path: str) -> Dict[str, Any]:
    """Read an Inspect log file regardless of .eval or .json format.

    Uses `inspect log dump` for .eval, else reads JSON directly.
    See Inspect docs:
    https://inspect.aisi.org.uk/eval-logs.html
    """
    if path.endswith(".eval"):
        import subprocess  # local import to avoid hard dep at import time

        proc = subprocess.run(
            ["inspect", "log", "dump", path],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_log_files(logfile: Optional[str], start_time: float) -> List[str]:
    """Collect log files created by this run.

    Preference:
    - if `logfile` provided and exists, return it
    - else scan INSPECT_LOG_DIR or ./logs for files
      with mtime >= start_time
    """
    candidates: List[str] = []

    if logfile and os.path.exists(logfile):
        return [os.path.abspath(logfile)]

    log_dir = os.getenv("INSPECT_LOG_DIR") or os.path.join(os.getcwd(), "logs")
    if not os.path.isdir(log_dir):
        return candidates

    for name in os.listdir(log_dir):
        if not (name.endswith(".eval") or name.endswith(".json")):
            continue
        path = os.path.join(log_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        recent = mtime >= (start_time - 1.0)
        if recent:
            candidates.append(os.path.abspath(path))

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def _flatten_results(data: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", {})
    out: Dict[str, Any] = {**base}
    out["total_samples"] = results.get("total_samples")
    out["completed_samples"] = results.get("completed_samples")
    scores = results.get("scores", [])
    if scores:
        metrics = scores[0].get("metrics", {})
        for metric_name, metric in metrics.items():
            out[metric_name] = metric.get("value")
    return out


def _flatten_stats(data: Dict[str, Any], base: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = data.get("stats", {})
    started_at = stats.get("started_at")
    completed_at = stats.get("completed_at")
    model_usage = stats.get("model_usage", {})
    rows: List[Dict[str, Any]] = []
    if isinstance(model_usage, dict):
        for model_name, usage in model_usage.items():
            row = {**base, "started_at": started_at}
            row["completed_at"] = completed_at
            row["usage_model"] = model_name
            row["input_tokens"] = usage.get("input_tokens")
            row["output_tokens"] = usage.get("output_tokens")
            # split for linter line length
            row["total_tokens"] = usage.get("total_tokens")
            rows.append(row)
    else:
        short = {
            **base,
            "started_at": started_at,
            "completed_at": completed_at,
        }
        rows.append(short)
    return rows


def _flatten_samples(
    data: Dict[str, Any], base: Dict[str, Any]
) -> List[Dict[str, Any]]:
    samples = data.get("samples", [])
    rows: List[Dict[str, Any]] = []
    for s in samples:
        row: Dict[str, Any] = {
            **base,
            "sample_id": s.get("id"),
            "epoch": s.get("epoch"),
            "target": s.get("target"),
            "messages": json.dumps(s.get("messages", [])),
        }
        if isinstance(s.get("metadata"), dict):
            for k, v in s["metadata"].items():
                row[f"meta_{k}"] = v
        if isinstance(s.get("scores"), dict):
            for scorer_name, score in s["scores"].items():
                row[f"score_{scorer_name}_value"] = score.get("value")
                row[f"score_{scorer_name}_answer"] = score.get("answer")
        rows.append(row)
    return rows


def export_logs_to_hub(
    *,
    logfile: Optional[str],
    start_time: float,
    hub_repo: str,
    hub_private: Optional[bool],
) -> None:
    files = _collect_log_files(logfile=logfile, start_time=start_time)
    if not files:
        msg = "No eval logs found to export (looked in INSPECT_LOG_DIR or ./logs)"
        typer.secho(msg, fg=typer.colors.YELLOW)
        return

    msg = f"Exporting {len(files)} eval logs to {hub_repo}"
    typer.secho(msg, fg=typer.colors.YELLOW)

    results_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []
    samples_rows: List[Dict[str, Any]] = []

    for path in files:
        try:
            data = _read_log_json(path)
        except Exception as e:  # pragma: no cover
            msg = f"Skipping log '{path}': {e}"
            typer.secho(msg, fg=typer.colors.YELLOW)
            continue

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

    if results_rows:
        ds = Dataset.from_list(results_rows)
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name="results",
            split="train",
            private=hub_private,
        )
        msg = (
            f"Pushed results ({len(results_rows)} rows) to {hub_repo} [config=results]"
        )
        typer.echo(msg)

    if stats_rows:
        ds = Dataset.from_list(stats_rows)
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name="stats",
            split="train",
            private=hub_private,
        )
        msg = f"Pushed stats ({len(stats_rows)} rows) to {hub_repo} [config=stats]"
        typer.echo(msg)

    if samples_rows:
        ds = Dataset.from_list(samples_rows)
        ds.push_to_hub(
            repo_id=hub_repo,
            config_name="samples",
            split="train",
            private=hub_private,
        )
        msg = (
            f"Pushed samples ({len(samples_rows)} rows) to {hub_repo} [config=samples]"
        )
        typer.echo(msg)
