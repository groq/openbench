import asyncio
import logging
import textwrap
import uuid

from inspect_ai.tool import Tool, tool
from inspect_ai.util import ExecResult, sandbox, store
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

TOOL_RESULT_MAX_CHARS = 3000
STDOUT_FILE = "/tmp/stdout.txt"
STDERR_FILE = "/tmp/stderr.txt"
CONTAINER_LAST_DIR_CACHE = "/tmp/bash_tool_last_dir"
CMD_WRAPPER = textwrap.dedent("""
    finally() {{
        pwd > {container_last_dir_cache}
        export -p > /tmp/bash_tool_last_env
    }}
    trap 'finally' EXIT

    if [ -f /tmp/bash_tool_last_env ]; then
        source /tmp/bash_tool_last_env &> /dev/null
    fi

    cd {cwd}

    exec 1> >(tee -a {stdout_file})
    exec 2> >(tee -a {stderr_file} >&2)

    {code}
    """).strip()


class Terminal(BaseModel):
    cwd: str = "~"  # Current working directory
    timeout_seconds: int = 600

    async def run_bash_command(self, command: str) -> ExecResult[str]:
        print(f"Running bash command: {command}")
        bash_sandbox = sandbox()

        # Prepare the files
        setup_command = (
            f"rm -f {STDOUT_FILE} {STDERR_FILE} && touch {STDOUT_FILE} {STDERR_FILE}"
        )
        await bash_sandbox.exec(["bash", "-c", setup_command])

        # Prepare the full command. We run the bash command in the given directory, and
        # then store the final directory in a file. Outputs are stored in temp files.
        wrapped_command = CMD_WRAPPER.format(
            cwd=self.cwd,
            code=command,
            container_last_dir_cache=CONTAINER_LAST_DIR_CACHE,
            stdout_file=STDOUT_FILE,
            stderr_file=STDERR_FILE,
        )

        try:
            result = await bash_sandbox.exec(
                ["bash", "-c", wrapped_command], timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            # Read output files
            try:
                stdout_content = await bash_sandbox.read_file(STDOUT_FILE)
            except Exception:
                stdout_content = ""
            try:
                stderr_content = await bash_sandbox.read_file(STDERR_FILE)
            except Exception:
                stderr_content = ""
            result = ExecResult(
                False,
                124,  # Timeout exit code
                stdout=stdout_content,
                stderr=stderr_content
                + ("\n\n" if stderr_content else "")
                + f"Warning: Your command timed out. Current timeout is set to "
                f"{self.timeout_seconds} seconds.",
            )

        # Remove output files
        await bash_sandbox.exec(["rm", "-f", STDOUT_FILE, STDERR_FILE])

        try:
            self.cwd = (
                await bash_sandbox.read_file(str(CONTAINER_LAST_DIR_CACHE))
            ).strip()
        except FileNotFoundError:
            LOGGER.info(
                f"Could not find the last directory file at {CONTAINER_LAST_DIR_CACHE}, "
                f"after running the following code:\n\n{command}"
            )

        return result


@tool
def run_bash_command() -> Tool:
    async def execute(command: str) -> str:
        """Run a command using bash. Returns the output of the command.

        Notes on command execution:
        - Commands are run in a stateless manner, but cwd and environment variables are maintained between calls.
        - Interactive commands aren't supported and will timeout.
        - Both stdout and stderr will be returned.

        Best practices:
        - Use grep or similar tools to search large files instead of cat (especially when reading truncated output files).
        - Avoid commands that produce very large outputs.
        - Check cwd if having path-related issues.

        :param command: The bash command to run. Provide a single command or multiple commands chained together. Avoid interactive commands. Be mindful of output size and use appropriate tools for large files.
        """
        terminal = store().get("terminal")
        assert isinstance(terminal, Terminal)
        result = await terminal.run_bash_command(command=command)
        return await _format_exec_result_for_model(result)

    return execute


async def _format_exec_result_for_model(
    result: ExecResult[str],
    save_full_output: bool = True,
    end_truncate: bool = False,
    visible_chars: int | None = None,
) -> str:
    """Format command execution result, optionally saving full output when truncated.

    Args:
        result: Execution result to format
        save_full_output: Whether to save full output to file when truncated
        end_truncate: If True, truncates at end instead of middle
        visible_chars: Override for maximum visible characters in output
    """
    # Calculate max chars based on visible_chars if provided
    max_chars = visible_chars if visible_chars is not None else TOOL_RESULT_MAX_CHARS

    # Only omit stderr and return code if both are empty/0
    if not result.stderr and not result.returncode:
        truncated_stdout, stdout_file = await _middle_truncate_string(
            result.stdout, max_chars, save_full_output, end_truncate, visible_chars
        )
        return truncated_stdout if truncated_stdout else "<No tool output>"

    elif result.stderr and not result.stdout:
        truncated_stderr, stderr_file = await _middle_truncate_string(
            result.stderr, max_chars, save_full_output, end_truncate, visible_chars
        )
        return f"# returncode:\n{result.returncode}\n\n# stdout:\n\n# stderr:\n{truncated_stderr}"

    # Handle case where both stdout and stderr are present
    # Split the visible chars between stdout and stderr if specified
    half_max_chars = max_chars // 2
    half_visible_chars = visible_chars // 2 if visible_chars is not None else None

    truncated_stdout, stdout_file = await _middle_truncate_string(
        result.stdout,
        half_max_chars,
        save_full_output,
        end_truncate,
        half_visible_chars,
    )
    truncated_stderr, stderr_file = await _middle_truncate_string(
        result.stderr,
        half_max_chars,
        save_full_output,
        end_truncate,
        half_visible_chars,
    )

    return f"# returncode:\n{result.returncode}\n\n# stdout:\n{truncated_stdout}\n\n# stderr:\n{truncated_stderr}"


async def _middle_truncate_string(
    input: str,
    max_length: int,
    save_full_output: bool = False,
    end_truncate: bool = False,
    visible_chars: int | None = None,
) -> tuple[str, str | None]:
    """Truncates the input string if it is longer than max_length.

    Args:
        input: String to truncate
        max_length: Maximum length of output string
        save_full_output: Whether to save full output to file
        end_truncate: If True, truncates at end instead of middle
        visible_chars: Override for maximum visible characters in output
    """
    if len(input) <= max_length:
        return input, None

    file_path = None
    actual_max_length = visible_chars if visible_chars is not None else max_length

    if save_full_output:
        file_path = await _save_output(input)
        truncation_message = f"\n<the output has been truncated to {actual_max_length} characters. Full output saved to {file_path}>\n"
    else:
        truncation_message = (
            f"\n<the output has been truncated to {actual_max_length} characters>\n"
        )

    assert len(truncation_message) <= max_length, (
        "max_length is too low to include truncation message"
    )

    if end_truncate:
        # Keep the beginning of the string and add truncation message at the end
        available_chars = actual_max_length - len(truncation_message)
        return input[:available_chars] + truncation_message, file_path
    else:
        # Original middle truncation logic
        start = (actual_max_length - len(truncation_message)) // 2
        end = actual_max_length - (start + len(truncation_message))
        return input[:start] + truncation_message + input[-end:], file_path


async def _save_output(content: str) -> str:
    """Save output to a file and return the file path."""
    file_id = str(uuid.uuid4())[:8]
    file_path = f"/shared/truncated_outputs/{file_id}.txt"

    await sandbox().write_file(file_path, content)

    return file_path
