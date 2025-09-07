"""
Roo-Code agent solver for interactive code development in Docker containers.

This solver creates a proper agent evaluation where the model operates inside
a Docker environment, interacts with files, runs commands, and iteratively
works toward passing tests - just like a real developer would.
"""

from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.util import sandbox
from inspect_ai.tool import Tool, tool


# Language-specific setup commands and environments
LANGUAGE_ENVIRONMENTS = {
    "python": {
        "image": "python:3.11-slim",
        "setup_commands": [
            # Python is already available in the base image
        ],
        "shell": "bash",
    },
    "go": {
        "image": "golang:1.21-alpine",
        "setup_commands": [
            "apt-get update",
            "apt-get install -y --no-install-recommends golang-go",
            "rm -rf /var/lib/apt/lists/*",
        ],
        "shell": "bash",
    },
    "javascript": {
        "image": "node:18-alpine",
        "setup_commands": [
            "curl -fsSL https://deb.nodesource.com/setup_18.x | bash -",
            "apt-get install -y nodejs",
            "npm install -g npm@latest",
        ],
        "shell": "bash",
    },
    "java": {
        "image": "openjdk:17-alpine",
        "setup_commands": [
            "apt-get update",
            "apt-get install -y --no-install-recommends openjdk-17-jdk",
            "rm -rf /var/lib/apt/lists/*",
        ],
        "shell": "bash",
    },
    "rust": {
        "image": "rust:1.75-alpine",
        "setup_commands": [
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            "source $HOME/.cargo/env",
            'export PATH="$HOME/.cargo/bin:$PATH"',
        ],
        "shell": "bash",
    },
}


def create_run_command_tool(task_dir: str, language: str) -> Tool:
    """
    Create a run_command tool with proper working directory context.

    Args:
        task_dir: The task directory path (e.g., "/workspace/python/task_name")
        language: Programming language for environment setup
    """

    @tool
    def run_command() -> Tool:
        """
        Tool for running shell commands in the Docker environment.
        The agent can use this to interact with the file system and run tests.
        """

        async def execute(command: str) -> str:
            """
            Execute a shell command in the task directory.

            Args:
                command: Shell command to execute

            Returns:
                Command output (stdout + stderr)
            """
            try:
                # Ensure we're in the correct directory before running the command
                full_command = f"cd {task_dir} && {command}"
                result = await sandbox().exec(
                    cmd=["bash", "-c", full_command],
                    timeout=30,
                )

                output_parts = []
                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    output_parts.append(f"STDERR:\n{result.stderr}")
                output_parts.append(f"EXIT_CODE: {result.returncode}")

                return "\n".join(output_parts)

            except Exception as e:
                return f"ERROR: Failed to execute command '{command}': {str(e)}"

        return execute

    return run_command()


def create_read_file_tool(task_dir: str) -> Tool:
    """
    Create a read_file tool with proper working directory context.

    Args:
        task_dir: The task directory path (e.g., "/workspace/python/task_name")
    """

    @tool
    def read_file() -> Tool:
        """
        Tool for reading file contents in the Docker environment.
        """

        async def execute(file_path: str) -> str:
            """
            Read the contents of a file.

            Args:
                file_path: Path to the file to read (relative to task directory)

            Returns:
                File contents or error message
            """
            try:
                # Handle both absolute and relative paths
                if not file_path.startswith("/"):
                    full_path = f"{task_dir}/{file_path}"
                else:
                    full_path = file_path

                result = await sandbox().exec(
                    cmd=["cat", full_path],
                    timeout=10,
                )

                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"ERROR: Could not read file '{file_path}': {result.stderr}"

            except Exception as e:
                return f"ERROR: Failed to read file '{file_path}': {str(e)}"

        return execute

    return read_file()


def create_write_file_tool(task_dir: str) -> Tool:
    """
    Create a write_file tool with proper working directory context.

    Args:
        task_dir: The task directory path (e.g., "/workspace/python/task_name")
    """

    @tool
    def write_file() -> Tool:
        """
        Tool for writing file contents in the Docker environment.
        """

        async def execute(file_path: str, content: str) -> str:
            """
            Write content to a file.

            Args:
                file_path: Path to the file to write (relative to task directory)
                content: Content to write to the file

            Returns:
                Success message or error
            """
            try:
                # Handle both absolute and relative paths
                if not file_path.startswith("/"):
                    full_path = f"{task_dir}/{file_path}"
                else:
                    full_path = file_path

                # Escape content for shell
                escaped_content = content.replace("'", "'\"'\"'")

                result = await sandbox().exec(
                    cmd=["bash", "-c", f"echo '{escaped_content}' > '{full_path}'"],
                    timeout=10,
                )

                if result.returncode == 0:
                    return f"Successfully wrote to '{file_path}'"
                else:
                    return f"ERROR: Could not write to '{file_path}': {result.stderr}"

            except Exception as e:
                return f"ERROR: Failed to write to file '{file_path}': {str(e)}"

        return execute

    return write_file()


def create_list_files_tool(task_dir: str) -> Tool:
    """
    Create a list_files tool with proper working directory context.

    Args:
        task_dir: The task directory path (e.g., "/workspace/python/task_name")
    """

    @tool
    def list_files() -> Tool:
        """
        Tool for listing files and directories.
        """

        async def execute(directory_path: str = ".") -> str:
            """
            List files in a directory.

            Args:
                directory_path: Path to directory to list (default: current task directory)

            Returns:
                Directory listing
            """
            try:
                # Handle both absolute and relative paths
                if directory_path == ".":
                    full_path = task_dir
                elif not directory_path.startswith("/"):
                    full_path = f"{task_dir}/{directory_path}"
                else:
                    full_path = directory_path

                result = await sandbox().exec(
                    cmd=["ls", "-la", full_path],
                    timeout=10,
                )

                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"ERROR: Could not list directory '{directory_path}': {result.stderr}"

            except Exception as e:
                return f"ERROR: Failed to list directory '{directory_path}': {str(e)}"

        return execute

    return list_files()


@solver
def roocode_agent_solver() -> Solver:
    """
    Agent-based solver for Roo-Code tasks.

    This solver:
    1. Sets up a Docker environment with the Roo-Code repository
    2. Gives the model tools to interact with the environment
    3. Lets the model iteratively work on the task until tests pass
    4. Captures the final test results for scoring
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        Execute the agent-based Roo-Code solving workflow.
        """
        # Extract metadata
        language = state.metadata.get("language")
        task_name = state.metadata.get("task_name")
        test_command = state.metadata.get("test_command")

        if not all([language, task_name, test_command]):
            state.output.completion = f"ERROR: Missing required metadata - language: {language}, task_name: {task_name}, test_command: {test_command}"
            return state

        # Type assertions for mypy
        assert isinstance(language, str), "language must be a string"
        assert isinstance(task_name, str), "task_name must be a string"
        assert isinstance(test_command, str), "test_command must be a string"

        try:
            # Set up the Docker environment
            setup_success = await setup_agent_environment(language, task_name)
            if not setup_success:
                state.output.completion = (
                    f"ERROR: Failed to set up environment for {language}/{task_name}"
                )
                return state

            # Construct the task directory path
            task_dir = f"/workspace/{language}/{task_name}"

            # Set up tools for the agent with proper working directory context
            state.tools = [
                create_run_command_tool(task_dir, language),
                create_read_file_tool(task_dir),
                create_write_file_tool(task_dir),
                create_list_files_tool(task_dir),
            ]

            # Let the agent work iteratively
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                # Generate agent response
                state = await generate(state)

                if not state.output or not state.output.completion:
                    break

                # Check if agent thinks it's done
                if "TASK_COMPLETE" in state.output.completion:
                    break

                iteration += 1

            # Run final test to capture results
            final_test_result = await run_final_test(test_command, task_dir, language)

            # Append final test results to completion
            state.output.completion += f"\n\n[FINAL_TEST_RESULTS]\n{final_test_result}"

        except Exception as e:
            state.output.completion = f"ERROR: Agent execution failed: {str(e)}"

        return state

    return solve


async def setup_agent_environment(language: str, task_name: str) -> bool:
    """
    Set up the Docker environment for the agent to work in.

    Args:
        language: Programming language
        task_name: Task name

    Returns:
        True if setup successful, False otherwise
    """
    if language not in LANGUAGE_ENVIRONMENTS:
        return False

    env_config = LANGUAGE_ENVIRONMENTS[language]

    # Setup commands to download the repository and install language tools
    commands = []

    # First: Create workspace and download repository
    commands.append("mkdir -p /workspace")
    commands.append(
        "git clone https://github.com/RooCodeInc/Roo-Code-Evals.git /workspace"
    )

    # Add language-specific setup commands if any
    if env_config["setup_commands"]:
        commands.extend(env_config["setup_commands"])

    # Final navigation to task directory
    commands.append(f"cd /workspace/{language}/{task_name}")

    # Execute setup
    full_command = " && ".join(commands)

    try:
        result = await sandbox().exec(
            cmd=["bash", "-c", full_command],
            timeout=300,
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        return result.returncode == 0
    except Exception as e:
        print(f"[DEBUG] Setup exception: {str(e)}")
        return False


async def run_final_test(test_command: str, task_dir: str, language: str) -> str:
    """
    Run the final test command and capture results.

    Args:
        test_command: Command to run tests
        task_dir: Task directory path
        language: Programming language for environment setup

    Returns:
        Test results string
    """
    try:
        # Ensure we're in the correct directory before running the test
        full_command = f"cd {task_dir} && {test_command}"
        result = await sandbox().exec(
            cmd=["bash", "-c", full_command],
            timeout=60,
        )

        output_parts = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]

        if result.stdout:
            output_parts.extend(["", "--- STDOUT ---", result.stdout])

        if result.stderr:
            output_parts.extend(["", "--- STDERR ---", result.stderr])

        return "\n".join(output_parts)

    except Exception as e:
        return f"ERROR: Failed to run final test: {str(e)}"
