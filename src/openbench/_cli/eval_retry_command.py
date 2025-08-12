from typing import List, Optional, Annotated
import sys
import typer
from inspect_ai import eval_retry
from inspect_ai.log._file import log_file_info
from inspect_ai._util.file import filesystem
from openbench.monkeypatch.display_results_patch import patch_display_results


def run_eval_retry(
    log_files: Annotated[
        List[str],
        typer.Argument(help="Log file(s) to retry failed evaluations from"),
    ],
    max_connections: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of parallel requests to the model",
            envvar="BENCH_MAX_CONNECTIONS",
        ),
    ] = None,
    max_subprocesses: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of parallel subprocesses",
            envvar="BENCH_MAX_SUBPROCESSES",
        ),
    ] = None,
    fail_on_error: Annotated[
        Optional[float],
        typer.Option(
            help="Failure threshold for sample errors. If between 0 and 1, it is interpreted as a percentage of samples that can fail. If greater than 1, it is interpreted as a fixed number of samples that can fail",
            envvar="BENCH_FAIL_ON_ERROR",
        ),
    ] = None,
    log_samples: Annotated[
        Optional[bool],
        typer.Option(
            help="Log detailed samples and scores",
            envvar="BENCH_LOG_SAMPLES",
        ),
    ] = None,
    log_images: Annotated[
        Optional[bool],
        typer.Option(
            help="Log base64 encoded images",
            envvar="BENCH_LOG_IMAGES",
        ),
    ] = None,
    log_buffer: Annotated[
        Optional[int],
        typer.Option(
            help="Number of samples to buffer before writing to log",
            envvar="BENCH_LOG_BUFFER",
        ),
    ] = 10,
    score: Annotated[
        bool,
        typer.Option(
            help="Grade the benchmark, or leave unscored",
            envvar="BENCH_SCORE",
        ),
    ] = True,
    timeout: Annotated[
        Optional[int],
        typer.Option(
            help="Timeout for each request to the model API in seconds",
            envvar="BENCH_TIMEOUT",
        ),
    ] = None,
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
    no_fail_on_error: Annotated[
        bool,
        typer.Option(
            "--no-fail-on-error",
            help="Do not fail the eval if errors occur within samples (instead, continue running other samples)",
            envvar="BENCH_NO_FAIL_ON_ERROR",
        ),
    ] = False,
    no_log_samples: Annotated[
        bool,
        typer.Option(
            "--no-log-samples",
            help="Do not include samples in the log file",
            envvar="BENCH_NO_LOG_SAMPLES",
        ),
    ] = False,
    no_log_images: Annotated[
        bool,
        typer.Option(
            "--no-log-images",
            help="Do not include base64 encoded images in the log file",
            envvar="BENCH_NO_LOG_IMAGES",
        ),
    ] = False,
    no_score: Annotated[
        bool,
        typer.Option(
            "--no-score",
            help="Do not score model output (use the inspect score command to score output later)",
            envvar="BENCH_NO_SCORE",
        ),
    ] = False,
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
    trace: Annotated[
        bool,
        typer.Option(
            "--trace",
            help="Trace message interactions with evaluated model to terminal",
            envvar="BENCH_TRACE",
        ),
    ] = False,
    log_dir: Annotated[
        str,
        typer.Option(
            help="Directory for log files",
            envvar="BENCH_LOG_DIR",
        ),
    ] = "./logs",
    debug_errors: Annotated[
        bool,
        typer.Option(
            "--debug-errors",
            help="Enable debug mode for errors",
            envvar="BENCH_DEBUG_ERRORS",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode with full stack traces",
            envvar="BENCH_DEBUG",
        ),
    ] = False,
) -> None:
    """Retry failed evaluation(s) from log files."""

    # Process negating options
    if no_log_samples:
        log_samples = False
    if no_log_images:
        log_images = False
    if no_score:
        score = False
    if no_sandbox_cleanup:
        sandbox_cleanup = False

    # Process fail_on_error
    if no_fail_on_error:
        fail_on_error = False
    elif fail_on_error == 0.0:
        fail_on_error = True

    # Process retry_on_error
    if retry_on_error == 0:
        retry_on_error = None

    # Resolve log files
    retry_log_files = [
        log_file_info(filesystem(log_file).info(log_file)) for log_file in log_files
    ]

    # Set defaults
    log_level = "info"
    log_level_transcript = "info"

    # Apply display patch
    patch_display_results()

    try:
        # Retry
        eval_retry(
            retry_log_files,
            log_level=log_level,
            log_level_transcript=log_level_transcript,
            log_dir=log_dir,
            max_connections=max_connections,
            max_subprocesses=max_subprocesses,
            fail_on_error=fail_on_error,
            retry_on_error=retry_on_error,
            debug_errors=debug_errors,
            log_samples=log_samples,
            log_images=log_images,
            log_buffer=log_buffer,
            score=score,
            timeout=timeout,
            max_retries=max_retries,
            sandbox_cleanup=sandbox_cleanup,
            trace=trace,
            # These are additional retry-specific parameters
            max_samples=None,
            max_tasks=None,
            max_sandboxes=None,
            log_shared=None,
            score_display=None,
        )

        typer.echo("Retry evaluation complete!")
    except Exception as e:
        if debug:
            # In debug mode, let the full stack trace show
            raise
        else:
            # In normal mode, show clean error message
            error_msg = str(e)
            typer.secho(f"\n❌ Error: {error_msg}", fg=typer.colors.RED, err=True)
            typer.secho(
                "\nFor full stack trace, run with --debug flag",
                fg=typer.colors.CYAN,
                err=True,
            )
            sys.exit(1)
