from typing import Optional, List, Dict, Annotated, Tuple, Union
from rich.console import Console
from enum import Enum
import sys
import time
import os
import typer
import asyncio
from inspect_ai import eval
from inspect_ai.model import Model
from inspect_ai.log import EvalLog
from openbench.config import load_task, EVAL_GROUPS
from openbench.monkeypatch.display_results_patch import patch_display_results
from openbench._cli.utils import parse_cli_args
from openbench.agents import AgentManager
from openbench.utils.livemcpbench_cache import (
    prepare_livemcpbench_cache,
    clear_livemcpbench_root,
)


class SandboxType(str, Enum):
    """Type of environment to run evaluations in."""

    LOCAL = "local"
    DOCKER = "docker"


class DisplayType(str, Enum):
    """Display type for evaluation progress."""

    FULL = "full"
    CONVERSATION = "conversation"
    RICH = "rich"
    PLAIN = "plain"
    NONE = "none"


class ReasoningEffortLevel(str, Enum):
    """Reasoning effort level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LogFormat(str, Enum):
    """Output format for benchmark logs."""

    EVAL = "eval"
    JSON = "json"


def parse_limit(value: Optional[str]) -> Optional[Union[int, Tuple[int, int]]]:
    """Parse the limit parameter which can be an int or a tuple of ints.

    Args:
        value: The value passed to the --limit option.

    Returns:
        Parsed limit value: int, tuple of (start, end), or None.

    Raises:
        typer.BadParameter: If the input format is incorrect.
    """
    if value is None:
        return None

    try:
        if "," in value:
            start, end = map(int, value.split(","))
            return (start, end)
        return int(value)
    except ValueError:
        raise typer.BadParameter(
            "Limit must be an integer or two integers separated by a comma"
        )


def validate_model_name(model: str, context: str = "") -> None:
    """Validate a model name format.

    Args:
        model: Model name to validate
        context: Additional context for error message

    Raises:
        typer.BadParameter: If model name format is invalid
    """
    if "/" not in model:
        raise typer.BadParameter(
            f"Invalid model name format{context}: {model}. Expected format: provider/model-name"
        )

    provider, remainder = model.split("/", 1)
    if not provider or not remainder:
        raise typer.BadParameter(
            f"Invalid model name format{context}: {model}. Expected format: provider/model-name"
        )


def validate_model_role(model_role: Optional[str]) -> Dict[str, str | Model]:
    """Validate and parse model role string.

    Args:
        model_role: Optional string in format 'role=model'

    Returns:
        Dictionary mapping role to model name

    Raises:
        typer.BadParameter: If model_role format is invalid
    """
    if not model_role:
        return {}

    try:
        role, model = model_role.split("=")
        if not role or not model:
            raise ValueError("Model role must be in format 'role=model'")
        validate_model_name(model, f" for role '{role}'")
        return {role: model}
    except ValueError as e:
        raise typer.BadParameter(str(e))


def expand_eval_groups(
    benchmarks: List[str],
) -> Tuple[List[str], dict[str, List[str]]]:
    """Expand eval group identifiers into their constituent benchmarks.

    Group names are normalized to handle - and _ interchangeably
    (e.g., "bigbench-lite" and "bigbench_lite" both work).

    Args:
        benchmarks: List of benchmark names and/or group identifiers

    Returns:
        Tuple of (all_expanded_benchmarks, groups):
        - all_expanded_benchmarks: All benchmarks including groups and individual tasks
        - groups: Dict mapping group display name to list of benchmark names

    Example:
        expand_eval_groups(["bigbench", "mmlu"])
        -> (["bigbench_arithmetic", "bigbench_...", "mmlu"], {"BIG-Bench": ["bigbench_arithmetic", ...]})
    """
    all_expanded = []
    groups = {}  # group_name -> list of benchmark names

    for benchmark in benchmarks:
        # Normalize: Try both with underscores and hyphens for compatibility
        # Users might type either cti_bench or cti-bench
        normalized_underscore = benchmark.replace("-", "_")
        normalized_hyphen = benchmark.replace("_", "-")

        # Check both normalized forms in EVAL_GROUPS
        group = None
        if normalized_underscore in EVAL_GROUPS:
            group = EVAL_GROUPS[normalized_underscore]
        elif normalized_hyphen in EVAL_GROUPS:
            group = EVAL_GROUPS[normalized_hyphen]

        if group:
            # Expand group to its constituent benchmarks
            typer.echo(
                f"📦 Expanding group '{benchmark}' -> {len(group.benchmarks)} benchmarks"
            )
            all_expanded.extend(group.benchmarks)
            # Store with display name as key
            groups[group.name] = group.benchmarks
        else:
            # Regular benchmark name (not a group)
            all_expanded.append(benchmark)

    return all_expanded, groups


def display_group_summary(
    group_name: str, group_benchmarks: List[str], eval_logs: List[EvalLog]
) -> None:
    """Display aggregate metrics for a single group.

    Computes mean accuracy, standard deviation, and standard error across benchmarks.
    Treats I/C scores as 1.0 (correct) and 0.0 (incorrect), and includes numerical scores
    on the 0.0-1.0 scale.

    Args:
        group_name: Display name of the group (e.g., "BIG-Bench", "BBH")
        group_benchmarks: List of benchmark names in this group
        eval_logs: List of evaluation logs from all benchmarks
    """
    import numpy as np

    # Filter to only logs from this group's benchmarks
    # Handle both 'benchmark' and 'openbench/benchmark' task name formats
    def task_matches_benchmark(task_name: str, benchmark_name: str) -> bool:
        """Check if task name matches benchmark, handling namespace prefixes."""
        # Strip namespace prefix if present (e.g., 'openbench/smt_algebra' -> 'smt_algebra')
        task_base = task_name.split("/")[-1] if "/" in task_name else task_name
        return task_base == benchmark_name

    group_logs = [
        log
        for log in eval_logs
        if any(
            task_matches_benchmark(log.eval.task, bench) for bench in group_benchmarks
        )
    ]

    if not group_logs:
        return

    # Collect per-benchmark accuracy scores (0.0-1.0 scale)
    benchmark_accuracies = []
    total_samples = 0

    for log in group_logs:
        if log.results:
            # Extract accuracy from EvalScore.metrics (correct API per inspect_ai)
            # log.results.scores is a list of EvalScore objects, each with a .metrics dict
            accuracy_value = None
            if log.results.scores:
                for score in log.results.scores:
                    if hasattr(score, "metrics") and isinstance(score.metrics, dict):
                        if "accuracy" in score.metrics:
                            metric = score.metrics["accuracy"]
                            accuracy_value = (
                                metric.value if hasattr(metric, "value") else metric
                            )
                            break

            # Include benchmarks with accuracy in aggregate calculation
            # Converts I/C to 1.0/0.0 and keeps numerical scores as-is
            if accuracy_value is not None:
                total_samples += log.results.completed_samples
                # Extract numeric value if it's an EvalMetric object
                numeric_value = float(
                    accuracy_value.value
                    if hasattr(accuracy_value, "value")
                    else accuracy_value
                )
                benchmark_accuracies.append(numeric_value)

    # Only display if we have data
    if len(benchmark_accuracies) == 0:
        # Debug: help users understand why no summary was shown
        if any(log.results for log in group_logs):
            typer.echo(
                "\n⚠️  Note: Group evaluation completed but aggregate summary unavailable. "
                "This may occur if benchmarks don't report accuracy metrics."
            )
        return

    # Calculate statistics
    mean_accuracy = float(np.mean(benchmark_accuracies))
    median_accuracy = float(np.median(benchmark_accuracies))
    std_accuracy = (
        float(np.std(benchmark_accuracies, ddof=1))
        if len(benchmark_accuracies) > 1
        else 0.0
    )
    stderr_accuracy = (
        std_accuracy / np.sqrt(len(benchmark_accuracies))
        if len(benchmark_accuracies) > 1
        else 0.0
    )

    # Calculate 95% confidence interval (mean ± 1.96 * stderr)
    ci_margin = 1.96 * stderr_accuracy if len(benchmark_accuracies) > 1 else 0.0
    ci_lower = max(0.0, mean_accuracy - ci_margin)  # Clamp to [0, 1]
    ci_upper = min(1.0, mean_accuracy + ci_margin)

    # Display summary
    typer.echo("\n" + "=" * 60)
    typer.echo(f"📊 GROUP SUMMARY - {group_name}")
    typer.echo("=" * 60)
    typer.echo(f"Total benchmarks:    {len(benchmark_accuracies)}")
    typer.echo(f"Total samples:       {total_samples:,}")
    typer.echo(f"Mean accuracy:       {mean_accuracy:.2%}")
    typer.echo(f"Median accuracy:     {median_accuracy:.2%}")
    typer.echo(f"95% CI:              [{ci_lower:.2%}, {ci_upper:.2%}]")
    typer.echo(f"Std deviation:       {std_accuracy:.4f}")
    typer.echo(f"Std error:           {stderr_accuracy:.4f}")
    typer.echo("=" * 60 + "\n")


def run_eval(
    benchmarks: Annotated[
        List[str],
        typer.Argument(
            help="Benchmark(s) to run. Can be a built-in name (e.g. mmlu), a group (e.g. bigbench, bbh, coding), or a path to a local eval directory/file containing __metadata__. Run 'bench list' to see all available groups.",
            envvar="BENCH_BENCHMARKS",
        ),
    ],
    model: Annotated[
        List[str],
        typer.Option(
            help="Model(s) to evaluate. Equivalent to --model-role candidate=<model>",
            envvar="BENCH_MODEL",
        ),
    ] = ["groq/openai/gpt-oss-20b"],
    max_connections: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of parallel requests to the model",
            envvar="BENCH_MAX_CONNECTIONS",
        ),
    ] = 10,
    model_base_url: Annotated[
        Optional[str],
        typer.Option(help="Base URL for model(s)", envvar="BENCH_MODEL_BASE_URL"),
    ] = None,
    model_role: Annotated[
        List[str],
        typer.Option(
            help="Model role(s). For example, --model-role grader=groq/meta-llama/llama-4-scout-17b-16e-instruct. Can be specified multiple times.",
            envvar="BENCH_MODEL_ROLE",
        ),
    ] = [],
    m: Annotated[
        List[str],
        typer.Option(
            "-M",
            help="One or more native model arguments (e.g. -M arg=value)",
            envvar="BENCH_MODEL_ARGS",
        ),
    ] = [],
    t: Annotated[
        List[str],
        typer.Option(
            "-T",
            help="One or more task arguments (e.g. -T arg=value)",
            envvar="BENCH_TASK_ARGS",
        ),
    ] = [],
    logfile: Annotated[
        Optional[str],
        typer.Option(help="Output file for results", envvar="BENCH_OUTPUT"),
    ] = None,
    sandbox: Annotated[
        Optional[SandboxType],
        typer.Option(
            help="Environment to run the evaluation in (local or docker)",
            case_sensitive=False,
            envvar="BENCH_SANDBOX",
        ),
    ] = None,
    sandbox_cleanup: Annotated[
        Optional[bool],
        typer.Option(
            help="Cleanup sandbox environments after task completes",
            envvar="BENCH_SANDBOX_CLEANUP",
        ),
    ] = None,
    no_sandbox_cleanup: Annotated[
        bool,
        typer.Option(
            "--no-sandbox-cleanup",
            help="Do not cleanup sandbox environments after task completes",
            envvar="BENCH_NO_SANDBOX_CLEANUP",
        ),
    ] = False,
    epochs: Annotated[
        int,
        typer.Option(
            help="Number of epochs to run each evaluation", envvar="BENCH_EPOCHS"
        ),
    ] = 1,
    limit: Annotated[
        Optional[str],
        typer.Option(
            help="Limit evaluated samples (single number or start,end)",
            envvar="BENCH_LIMIT",
        ),
    ] = None,
    fail_on_error: Annotated[
        Optional[float],
        typer.Option(
            help="Failure threshold for sample errors. If between 0 and 1, it is interpreted as a percentage of samples that can fail. If greater than 1, it is interpreted as a fixed number of samples that can fail",
            envvar="BENCH_FAIL_ON_ERROR",
        ),
    ] = None,
    no_fail_on_error: Annotated[
        bool,
        typer.Option(
            "--no-fail-on-error",
            help="Do not fail the eval if errors occur within samples (instead, continue running other samples)",
            envvar="BENCH_NO_FAIL_ON_ERROR",
        ),
    ] = False,
    message_limit: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of messages one sample can run",
            envvar="BENCH_MESSAGE_LIMIT",
        ),
    ] = None,
    max_subprocesses: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of parallel subprocesses",
            envvar="BENCH_MAX_SUBPROCESSES",
        ),
    ] = None,
    log_samples: Annotated[
        Optional[bool],
        typer.Option(
            help="Log detailed samples and scores",
            envvar="BENCH_LOG_SAMPLES",
        ),
    ] = None,
    no_log_samples: Annotated[
        bool,
        typer.Option(
            "--no-log-samples",
            help="Do not include samples in the log file",
            envvar="BENCH_NO_LOG_SAMPLES",
        ),
    ] = False,
    log_images: Annotated[
        Optional[bool],
        typer.Option(
            help="Log base64 encoded images",
            envvar="BENCH_LOG_IMAGES",
        ),
    ] = None,
    no_log_images: Annotated[
        bool,
        typer.Option(
            "--no-log-images",
            help="Do not include base64 encoded images in the log file",
            envvar="BENCH_NO_LOG_IMAGES",
        ),
    ] = False,
    log_buffer: Annotated[
        Optional[int],
        typer.Option(
            help="Number of samples to buffer before writing to log",
            envvar="BENCH_LOG_BUFFER",
        ),
    ] = 10,
    log_dir: Annotated[
        Optional[str],
        typer.Option(
            help="Directory for log files",
            envvar="BENCH_LOG_DIR",
        ),
    ] = "./logs",
    trace: Annotated[
        bool,
        typer.Option(
            "--trace",
            help="Trace message interactions with evaluated model to terminal",
            envvar="BENCH_TRACE",
        ),
    ] = False,
    debug_errors: Annotated[
        bool,
        typer.Option(
            "--debug-errors",
            help="Enable debug mode for errors",
            envvar="BENCH_DEBUG_ERRORS",
        ),
    ] = False,
    score: Annotated[
        bool,
        typer.Option(
            help="Grade the benchmark, or leave unscored",
            envvar="BENCH_SCORE",
        ),
    ] = True,
    no_score: Annotated[
        bool,
        typer.Option(
            "--no-score",
            help="Do not score model output (use the inspect score command to score output later)",
            envvar="BENCH_NO_SCORE",
        ),
    ] = False,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            help="Model temperature",
            envvar="BENCH_TEMPERATURE",
        ),
    ] = None,
    top_p: Annotated[
        Optional[float],
        typer.Option(
            help="Model top-p",
            envvar="BENCH_TOP_P",
        ),
    ] = None,
    max_tasks: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of tasks to run concurrently",
            envvar="BENCH_MAX_TASKS",
        ),
    ] = None,
    max_tokens: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum tokens for model response",
            envvar="BENCH_MAX_TOKENS",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option(
            help="Seed for deterministic generation",
            envvar="BENCH_SEED",
        ),
    ] = None,
    display: Annotated[
        Optional[DisplayType],
        typer.Option(
            help="Display type for evaluation progress",
            envvar="BENCH_DISPLAY",
            case_sensitive=False,
        ),
    ] = None,
    timeout: Annotated[
        Optional[int],
        typer.Option(
            help="Timeout for each request to the model API in seconds",
            envvar="BENCH_TIMEOUT",
        ),
    ] = 10000,
    max_retries: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of times to retry model API requests (defaults to unlimited)",
            envvar="BENCH_MAX_RETRIES",
        ),
    ] = None,
    retry_on_error: Annotated[
        Optional[int],
        typer.Option(
            help="Retry samples if they encounter errors (by default, no retries occur). Specify --retry-on-error to retry a single time, or specify e.g. --retry-on-error=3 to retry multiple times.",
            envvar="BENCH_RETRY_ON_ERROR",
        ),
    ] = None,
    reasoning_effort: Annotated[
        Optional[ReasoningEffortLevel],
        typer.Option(
            help="Reasoning effort level. used for reasoning models like openai/o3",
            envvar="BENCH_REASONING_EFFORT",
            case_sensitive=False,
        ),
    ] = None,
    log_format: Annotated[
        Optional[LogFormat],
        typer.Option(
            help="Output logging format",
            envvar="BENCH_LOG_FORMAT",
            case_sensitive=False,
        ),
    ] = LogFormat.EVAL,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode with full stack traces",
            envvar="BENCH_DEBUG",
        ),
    ] = False,
    hub_repo: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Target Hub dataset repo (e.g. username/openbench-logs). "
                "If provided, logs will be exported to this dataset"
            ),
            envvar="BENCH_HUB_REPO",
        ),
    ] = None,
    hub_private: Annotated[
        Optional[bool],
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
    keep_livemcp_root: Annotated[
        bool,
        typer.Option(
            "--keep-livemcp-root",
            help="Do not auto-clean ~/.openbench/livemcpbench/root after eval",
            envvar="BENCH_KEEP_LIVEMCP_ROOT",
        ),
    ] = False,
    alpha: Annotated[
        bool,
        typer.Option(
            "--alpha",
            help="Allow running experimental/alpha benchmarks",
            envvar="BENCH_ALPHA",
        ),
    ] = False,
    code_agent: Annotated[
        Optional[str],
        typer.Option(
            "--code-agent",
            help=AgentManager.get_help_text(),
            envvar="BENCH_CODE_AGENT",
        ),
    ] = None,
) -> List[EvalLog] | None:
    """
    Run a benchmark on a model.
    """
    # Parse model and task arguments
    model_args = parse_cli_args(m) if m else {}
    task_args = parse_cli_args(t) if t else {}

    # Add code agent to task arguments if specified
    if code_agent:
        if not AgentManager.validate_code_agent(code_agent):
            valid_agents = AgentManager.get_valid_code_agents()
            raise typer.BadParameter(
                f"Invalid code agent: {code_agent}. Valid options: {', '.join(valid_agents)}"
            )
        task_args["code_agent"] = code_agent

        # Override default model for code agent if still using default
        if model == ["groq/openai/gpt-oss-20b"]:
            default_model = AgentManager.get_default_model(code_agent)
            model = [default_model]

    # Validate and aggregate model_role(s) into a dict
    role_models = {}
    for mr in model_role:
        parsed = validate_model_role(mr)
        for k, v in parsed.items():
            if k in role_models:
                raise typer.BadParameter(f"Duplicate model role: {k}")
            role_models[k] = v

    # Check for mutual exclusivity between --model and --model-role candidate
    if model and "candidate" in role_models:
        raise typer.BadParameter(
            "Cannot specify both --model and --model-role candidate=<model>"
        )

    # If using Roo code agent, enforce OpenRouter models
    if code_agent and code_agent.lower() == "roo":
        for model_name in model:
            if not model_name.startswith("openrouter/"):
                raise typer.BadParameter(
                    "For --code-agent roo, --model must be an OpenRouter model id prefixed with 'openrouter/'. "
                    "Example: --model openrouter/anthropic/claude-sonnet-4-20250514"
                )

    # Validate model names
    for model_name in model:
        validate_model_name(model_name)

    # Expand eval groups into individual benchmarks
    expanded_benchmarks, groups = expand_eval_groups(benchmarks)

    # Load tasks from registry
    tasks = []
    for benchmark in expanded_benchmarks:
        try:
            task = load_task(benchmark, allow_alpha=alpha)
            tasks.append(task)
        except (ValueError, ImportError, AttributeError) as e:
            raise typer.BadParameter(str(e))

    try:
        # auto-prepare caches for livemcpbench
        if "livemcpbench" in expanded_benchmarks:
            prepare_livemcpbench_cache()
        # auto-prepare CVEBench challenges directory
        if "cvebench" in expanded_benchmarks:
            from importlib import import_module

            datasets = import_module("openbench_cyber.datasets.cvebench")
            plugin_dir = datasets._default_challenges_dir().resolve()
            os.environ["CVEBENCH_CHALLENGE_DIR"] = str(plugin_dir)
    except Exception as e:
        raise typer.BadParameter(str(e))

    # Monkey patch FileRecorder log file name if logfile is provided
    if logfile:
        from openbench.monkeypatch.file_recorder_logfile_patch import (
            patch_file_recorder_logfile,
        )

        patch_file_recorder_logfile(logfile)

    # Parse limit string to int or tuple
    parsed_limit = parse_limit(limit)

    # Apply display patch
    patch_display_results()

    # Capture start time to locate logs created by this run
    start_time = time.time()

    # Process negating options to mirror eval-retry behavior
    if no_log_samples:
        log_samples = False
    if no_log_images:
        log_images = False
    if no_score:
        score = False
    if no_sandbox_cleanup:
        sandbox_cleanup = False
    if no_fail_on_error:
        fail_on_error = False
    elif fail_on_error == 0.0:
        fail_on_error = True

    # Align retry semantics with eval-retry command
    if retry_on_error == 0:
        retry_on_error = None

    try:
        try:
            eval_logs = eval(
                tasks=tasks,
                model=model,
                max_connections=max_connections,
                model_base_url=model_base_url,
                model_args=model_args,
                model_roles=role_models if role_models else None,
                task_args=task_args,
                epochs=epochs,
                limit=parsed_limit,
                fail_on_error=fail_on_error,
                message_limit=message_limit,
                max_subprocesses=max_subprocesses,
                log_samples=log_samples,
                log_images=log_images,
                log_buffer=log_buffer,
                log_dir=log_dir,
                score=score,
                debug_errors=debug_errors,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                display=display.value if display else None,
                timeout=timeout,
                trace=trace,
                max_retries=max_retries,
                retry_on_error=retry_on_error,
                reasoning_effort=reasoning_effort.value if reasoning_effort else None,
                sandbox=sandbox,
                sandbox_cleanup=sandbox_cleanup,
                log_format=log_format.value if log_format else None,
                max_tasks=max_tasks,
            )

            typer.echo("Evaluation complete!")

            # Display group summary if groups were used
            # Display separate summary for each group
            if groups:
                for group_name, group_benchmarks in groups.items():
                    display_group_summary(group_name, group_benchmarks, eval_logs)

            if hub_repo:
                from openbench._cli.export import export_logs_to_hub

                export_logs_to_hub(
                    logfile=logfile,
                    start_time=start_time,
                    hub_repo=hub_repo,
                    hub_private=hub_private,
                    hub_benchmark_name=hub_benchmark_name,
                    hub_model_name=hub_model_name,
                )
            return eval_logs
        except Exception as e:
            if debug:
                raise
            else:
                # In normal mode, show clean error message
                console = Console(stderr=True)
                error_msg = str(e)
                console.print(f"\n[red bold]❌ {error_msg}[/red bold]")
                console.print(
                    "\n[cyan]For full stack trace, run with --debug flag[/cyan]"
                )
                sys.exit(1)
    finally:
        # Auto-clean root sandbox for livemcpbench unless opted out
        if "livemcpbench" in expanded_benchmarks and not keep_livemcp_root:
            clear_livemcpbench_root(quiet=False)
        if "factscore" in expanded_benchmarks:
            from openbench.scorers.factscore import cleanup_factscore_runners

            try:
                asyncio.run(cleanup_factscore_runners())
            except Exception:
                # Silently ignore cleanup errors
                pass
