"""
Monkey patch for inspect_ai display results to customize evaluation display.

Adds sample duration metrics (average, p95, p50) to the evaluation results display
by reading sample timing data from the evaluation log.

Usage:
    from openbench.monkeypatch.display_results_patch import patch_display_results
    patch_display_results()

Call this before invoking inspect_ai.eval().
"""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import RenderableType
    from inspect_ai.log import EvalStats


def patch_display_results():
    """
    Monkey patch inspect_ai display functions to customize evaluation results display.
    """
    try:
        import inspect_ai._display.core.results as results_mod
        from rich.table import Table
        from rich.text import Text

        # Store original functions
        original_task_interrupted = results_mod.task_interrupted

        def custom_task_interrupted(profile, samples_completed):  # type: ignore
            # Call original function
            result = original_task_interrupted(profile, samples_completed)

            # If result is a string, replace the text
            if isinstance(result, str):
                result = result.replace("inspect eval-retry", "bench eval-retry")
            # If it's a Text object from rich, we need to handle it differently
            elif hasattr(result, "_text") and isinstance(result._text, list):
                # Rich Text objects store segments internally
                for i, segment in enumerate(result._text):
                    if isinstance(segment, tuple) and len(segment) >= 1:
                        text = segment[0]
                        if isinstance(text, str) and "inspect eval-retry" in text:
                            # Create a new segment with replaced text
                            new_text = text.replace(
                                "inspect eval-retry", "bench eval-retry"
                            )
                            result._text[i] = (new_text,) + segment[1:]

            return result

        def custom_task_stats(
            stats: "EvalStats", log_location: str | None = None
        ) -> "RenderableType":
            from inspect_ai._display.core.rich import rich_theme
            import statistics

            theme = rich_theme()
            panel = Table.grid(expand=True)
            panel.add_column()
            if len(stats.model_usage) < 2:
                panel.add_row()

            table = Table.grid(expand=True)
            table.add_column(style="bold")
            table.add_column()

            # Eval time
            started = datetime.fromisoformat(stats.started_at)
            completed = datetime.fromisoformat(stats.completed_at)
            elapsed = completed - started
            table.add_row(
                Text("total time:", style="bold"), f"  {elapsed}", style=theme.light
            )

            # Token usage
            for model, usage in stats.model_usage.items():
                if (
                    usage.input_tokens_cache_read is not None
                    or usage.input_tokens_cache_write is not None
                ):
                    input_tokens_cache_read = usage.input_tokens_cache_read or 0
                    input_tokens_cache_write = usage.input_tokens_cache_write or 0
                    input_tokens = f"[bold]I: [/bold]{usage.input_tokens:,}, [bold]CW: [/bold]{input_tokens_cache_write:,}, [bold]CR: [/bold]{input_tokens_cache_read:,}"
                else:
                    input_tokens = f"[bold]I: [/bold]{usage.input_tokens:,}"

                if usage.reasoning_tokens is not None:
                    reasoning_tokens = f", [bold]R: [/bold]{usage.reasoning_tokens:,}"
                else:
                    reasoning_tokens = ""

                table.add_row(
                    Text(model, style="bold"),
                    f"  {usage.total_tokens:,} tokens [{input_tokens}, [bold]O: [/bold]{usage.output_tokens:,}{reasoning_tokens}]",
                    style=theme.light,
                )

            # Empty row for spacing
            table.add_row()

            # Calculate sample duration metrics from log
            avg_duration_str = "N/A"
            p95_str = "N/A"
            p50_str = "N/A"

            if log_location:
                try:
                    from inspect_ai.log import read_eval_log

                    log = read_eval_log(log_location)
                    if log and log.samples:
                        # Extract total_time from each sample where it exists
                        sample_durations = [
                            s.total_time
                            for s in log.samples
                            if s.total_time is not None
                        ]

                        if sample_durations:
                            avg_duration = statistics.mean(sample_durations)
                            p50_duration = statistics.median(sample_durations)

                            # Calculate 95th percentile using numpy's percentile function
                            import numpy as np

                            p95_duration = np.percentile(sample_durations, 95)

                            avg_duration_str = f"{avg_duration:.2f}s"
                            p50_str = f"{p50_duration:.2f}s"
                            p95_str = f"{p95_duration:.2f}s"
                except Exception:
                    # If we can't read timing data, use defaults
                    pass

            # Add sample duration metrics section
            table.add_row(
                Text("average sample duration:", style="bold"),
                f"  {avg_duration_str}",
                style=theme.light,
            )
            table.add_row(
                Text("p95 sample duration:", style="bold"),
                f"  {p95_str}",
                style=theme.light,
            )
            table.add_row(
                Text("p50 sample duration:", style="bold"),
                f"  {p50_str}",
                style=theme.light,
            )

            panel.add_row(table)
            return panel

        def custom_task_result_summary(profile, success):
            from inspect_ai._display.core.config import task_config
            from inspect_ai._display.core.panel import task_panel
            from inspect_ai._display.core.results import task_results

            # The contents of the panel
            config = task_config(profile)
            body = custom_task_stats(success.stats, profile.log_location)

            # The panel
            return task_panel(
                profile=profile,
                show_model=True,
                body=body,
                subtitle=config,
                footer=task_results(profile, success),
                log_location=profile.log_location,
            )

        # Apply patches
        results_mod.task_interrupted = custom_task_interrupted
        results_mod.task_result_summary = custom_task_result_summary

    except (ImportError, AttributeError):
        # If inspect_ai is not installed or the module structure changed, silently continue
        pass
