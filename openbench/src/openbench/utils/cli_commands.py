"""
Utility functions for CLI-based solvers that run tasks inside Docker sandboxes.

This module provides common functionality for different CLI code agents (aider, opencode, roo)
including repository management, environment setup, and command execution.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from inspect_ai.util import sandbox

from openbench.provider_config import ProviderManager


# =============================================================================
# Common Repository and Environment Management
# =============================================================================


async def ensure_repo_and_task(language: str, task_name: str) -> bool:
    """Clone exercism tasks into /workspace if needed and verify task exists.

    Args:
        language: Programming language (e.g., 'python', 'javascript')
        task_name: Name of the specific task

    Returns:
        True if setup successful, False otherwise
    """
    try:
        commands: List[str] = [
            "mkdir -p /workspace",
            # Clone only if not already present
            "[ -d /workspace/.git ] || git clone https://github.com/RooCodeInc/Roo-Code-Evals.git /workspace",
            f"test -d /workspace/{language}/{task_name}",
            f"ls -la /workspace/{language}/{task_name}",
        ]
        result = await sandbox().exec(
            cmd=["bash", "-lc", " && ".join(commands)],
            timeout=180,
        )
        return result.returncode == 0
    except Exception:
        return False


async def run_setup_commands(setup_commands: List[str], workdir: str) -> str:
    """Run optional language-specific setup commands inside the task directory.

    Args:
        setup_commands: List of shell commands to execute
        workdir: Working directory path for the task

    Returns:
        Formatted output string with results
    """
    if not setup_commands:
        return "No setup commands"

    joined = " && ".join(setup_commands)
    try:
        result = await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} && ({joined})"],
            timeout=900,
        )
        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])
        return "\n".join(parts)
    except Exception as e:
        return f"ERROR: setup failed: {e}"


async def run_final_test(test_command: str, workdir: str) -> str:
    """Run the final test command in the task directory and capture results.

    Args:
        test_command: Shell command to run tests
        workdir: Working directory path for the task

    Returns:
        Formatted output string with test results
    """
    try:
        # Fix Python test commands
        fixed_test_command = test_command
        if "python" in workdir.lower():
            fixed_test_command = re.sub(
                r"([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*)-test\.py",
                lambda m: m.group(0).replace("-", "_"),
                test_command,
            )
            fixed_test_command = re.sub(
                r"([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+)(_test\.py)",
                lambda m: m.group(1).replace("-", "_") + m.group(2),
                fixed_test_command,
            )

        result = await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} && {fixed_test_command}"],
            timeout=600,
        )
        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]

        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])
        return "\n".join(parts)
    except Exception as e:
        return f"ERROR: test run failed: {e}"


def get_provider_env_keys() -> List[str]:
    """Get the list of all provider environment variable keys.

    Returns:
        List of environment variable names for all supported providers
    """
    return ProviderManager.get_all_env_vars()


def collect_provider_env() -> Dict[str, str]:
    """Collect a comprehensive set of provider API keys from host environment.

    Defaults to empty strings when not present, so the sandbox always receives
    a consistent mapping without leaking missing envs as None.

    Returns:
        Dictionary mapping environment variable names to their values
    """
    return ProviderManager.get_env_vars_dict()


def generate_env_setup_script() -> str:
    """Generate bash script content to export all provider environment variables.

    Returns:
        Bash script content that exports all environment variables
    """
    keys = get_provider_env_keys()

    lines = [
        "# Set up environment variables for API access",
    ]

    for key in keys:
        lines.append(f'export {key}="${{{key}:-}}"')

    return "\n".join(lines)


async def write_prompt_to_file(prompt_text: str, filename: str) -> bool:
    """Write prompt to a temporary file to avoid shell quoting issues.

    Args:
        prompt_text: The prompt text to write
        filename: The filename to write to (in /tmp)

    Returns:
        True if successful, False otherwise
    """
    try:
        write_prompt = await sandbox().exec(
            cmd=[
                "bash",
                "-lc",
                f"cat > /tmp/{filename} <<'EOF'\n{prompt_text}\nEOF",
            ],
            timeout=15,
        )
        return write_prompt.returncode == 0
    except Exception:
        return False


async def write_and_execute_script(
    script_content: str,
    script_name: str,
    timeout: int = 1800,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Write a script to file and execute it.

    Args:
        script_content: The script content to write
        script_name: Name of the script file (in /tmp)
        timeout: Execution timeout in seconds
        env: Environment variables to pass

    Returns:
        Dictionary with 'returncode', 'stdout', 'stderr', and 'success' keys
    """
    try:
        # Write the script
        script_write = await sandbox().exec(
            cmd=[
                "bash",
                "-c",
                f"cat > /tmp/{script_name} <<'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF",
            ],
            timeout=30,
        )
        if script_write.returncode != 0:
            return {
                "returncode": script_write.returncode,
                "stdout": "",
                "stderr": f"Failed to write script: {script_write.stderr}",
                "success": False,
            }

        # Make script executable
        chmod_result = await sandbox().exec(
            cmd=["chmod", "+x", f"/tmp/{script_name}"],
            timeout=30,
        )
        if chmod_result.returncode != 0:
            return {
                "returncode": chmod_result.returncode,
                "stdout": "",
                "stderr": f"Failed to make script executable: {chmod_result.stderr}",
                "success": False,
            }

        # Execute the script
        result = await sandbox().exec(
            cmd=[f"/tmp/{script_name}"],
            timeout=timeout,
            env=env or collect_provider_env(),
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "success": result.returncode == 0,
        }

    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Script execution failed: {str(e)}",
            "success": False,
        }


async def read_log_file(
    log_path: str, log_name: str, tail_lines: Optional[int] = None
) -> str:
    """Read a log file from the sandbox.

    Args:
        log_path: Path to the log file
        log_name: Name for the log section
        tail_lines: Number of lines to tail (if specified)

    Returns:
        Formatted log output
    """
    try:
        cmd_parts = ["bash", "-lc"]
        if tail_lines:
            cmd_parts.append(
                f"tail -n {tail_lines} {log_path} || echo 'No {log_name.lower()} log found'"
            )
        else:
            cmd_parts.append(
                f"cat {log_path} || echo 'No {log_name.lower()} log found'"
            )

        log_read = await sandbox().exec(
            cmd=cmd_parts,
            timeout=10,
        )
        if log_read.stdout:
            return f"--- {log_name.upper()} LOG{' (tail)' if tail_lines else ''} ---\n{log_read.stdout}"
        else:
            return f"--- {log_name.upper()} LOG ---\nNo log content found"
    except Exception:
        return f"--- {log_name.upper()} LOG ---\nFailed to read log file"


def format_execution_output(
    result: Dict[str, Any], additional_logs: Optional[List[str]] = None
) -> str:
    """Format execution output consistently.

    Args:
        result: Result dictionary from execute_script
        additional_logs: List of additional log content to append

    Returns:
        Formatted output string
    """
    parts: List[str] = [
        f"Exit Code: {result['returncode']}",
        f"Success: {result['success']}",
    ]

    if result["stdout"]:
        parts.extend(["", "--- STDOUT ---", result["stdout"]])
    if result["stderr"]:
        parts.extend(["", "--- STDERR ---", result["stderr"]])

    if additional_logs:
        for log_content in additional_logs:
            if log_content:
                parts.extend(["", log_content])

    return "\n".join(parts)


# =============================================================================
# Script Templates for Code Agents
# =============================================================================


def get_aider_script_template() -> str:
    """Get the Aider execution script template.

    Returns:
        Aider script template with placeholders
    """
    return """#!/bin/bash
set +e

cd {workdir}

# Read the prompt from file
PROMPT=$(cat /tmp/aider_prompt.txt)

{env_setup}

MODEL_ARG="--model {model}"

# Run aider with the prompt, passing all files in the directory
echo "Running Aider with prompt: $PROMPT"
echo "Model: $MODEL_ARG"
echo "Working directory: $(pwd)"
echo "Directory contents:"
ls -la

# Find all files in the current directory (excluding hidden files, directories, and common non-source files)
ALL_FILES=$(find . -maxdepth 10 -type f ! -path '*/.*' ! -name '.*' ! -name '*.log' ! -name '*.tmp' ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path '*/node_modules/*' ! -path '*/venv/*' ! -path '*/.venv/*' | sort)

echo "Files to pass to Aider:"
echo "$ALL_FILES"

if [ -n "$ALL_FILES" ]; then
    # Convert newline-separated files to space-separated arguments
    aider $MODEL_ARG --no-auto-commit -m "$PROMPT" $ALL_FILES 2>&1 | tee /tmp/aider-output.log
else
    echo "No files found in directory, running aider on current directory"
    aider $MODEL_ARG --no-auto-commit -m "$PROMPT" . 2>&1 | tee /tmp/aider-output.log
fi
"""


def get_opencode_script_template() -> str:
    """Get the OpenCode execution script template.

    Returns:
        OpenCode script template with placeholders
    """
    return """#!/bin/bash
set +e

cd {workdir}

# Read the prompt from file  
PROMPT=$(cat /tmp/opencode_prompt.txt)

{env_setup}

echo "Running OpenCode with prompt: $PROMPT"
echo "Model: {model}"
echo "Working directory: $(pwd)"

opencode run -m {model} "$PROMPT" 2>&1 | tee /tmp/opencode-output.log
"""


def get_claude_script_template() -> str:
    """Get the Claude Code execution script template.

    Returns:
        Claude Code script template with placeholders
    """
    return """#!/bin/bash
set +e

cd {workdir}

# Read the prompt from file
PROMPT=$(cat /tmp/claude_code_prompt.txt)

{env_setup}

echo "Running Claude Code with prompt: $PROMPT"  
echo "Model: {model}"
echo "Working directory: $(pwd)"

echo "$PROMPT" | claude -p --model "{model}" \
    --permission-mode acceptEdits \
    --allowedTools "Bash(*)" "Read" "Edit" \
    2>&1 | tee /tmp/claude-code-output.log
"""


def get_roo_script_template() -> str:
    """Get the Roo CLI execution script template.

    Returns:
        Roo script template with placeholders
    """
    return """#!/bin/bash
set -eo pipefail

# ========= Environment Setup =========
export WORKDIR="{workdir}"
export VSCODE_EXT_DIR="/opt/vscode-extensions"
export VSCODE_USER_DIR="/opt/vscode-user"

# Save task prompt to file to avoid shell escaping issues
cat > /tmp/task_prompt.txt << 'TASK_PROMPT_EOF'
{enhanced_prompt}
TASK_PROMPT_EOF

{env_setup}

# Clean up any existing VS Code processes
echo "[INFO] Cleaning up existing VS Code processes..."
pkill -f "code.*${{VSCODE_USER_DIR}}" || true
sleep 2

# Create workspace settings for the specific task
mkdir -p "${{WORKDIR}}/.vscode"
cat > "${{WORKDIR}}/.vscode/settings.json" << 'VSCODE_SETTINGS_EOF'
{{
  "security.workspace.trust.enabled": false,
  "telemetry.telemetryLevel": "off",
  "extensions.autoUpdate": false,
  "roo-cline.autoImportSettingsPath": "/etc/roo/roo-code-settings.json"
}}
VSCODE_SETTINGS_EOF

echo "[INFO] Starting VS Code on task directory: ${{WORKDIR}}"
: > /tmp/code.log
xvfb-run -a env ROO_CODE_IPC_SOCKET_PATH="/tmp/roo-code.sock" \\
  code --no-sandbox --verbose --log trace --disable-workspace-trust --use-inmemory-secretstorage \\
    --extensions-dir "${{VSCODE_EXT_DIR}}" \\
    --user-data-dir "${{VSCODE_USER_DIR}}" \\
    "${{WORKDIR}}" >/tmp/code.log 2>&1 &
CODE_PID=$!

# ========= Wait for Roo Socket =========
export ROO_CODE_IPC_SOCKET_PATH="/tmp/roo-code.sock"
echo "Waiting for Roo socket at ${{ROO_CODE_IPC_SOCKET_PATH}}..."
for i in $(seq 1 120); do
  if [ -S "${{ROO_CODE_IPC_SOCKET_PATH}}" ]; then
    echo "Roo socket ready: ${{ROO_CODE_IPC_SOCKET_PATH}}"
    break
  fi
  sleep 1
done

if [ ! -S "${{ROO_CODE_IPC_SOCKET_PATH}}" ]; then
  echo "ERROR: Roo socket not created after 120 seconds"
  exit 2
fi

# ========= Setup roo-cli =========
cd /opt/roo-cli

# Update .env file with API configuration
if [ -n "$OPENROUTER_API_KEY" ] && [ "$OPENROUTER_API_KEY" != "" ]; then
    echo "OPENROUTER_API_KEY=${{OPENROUTER_API_KEY}}" >> .env
    echo "OPENAI_API_KEY=${{OPENROUTER_API_KEY}}" >> .env
    echo "OPENAI_BASE_URL=https://openrouter.ai/api/v1" >> .env
    echo "OPENAI_MODEL=anthropic/claude-3.5-sonnet" >> .env
else
    echo "ERROR: No OPENROUTER_API_KEY provided"
    echo "OPENROUTER_API_KEY=" >> .env
    echo "OPENAI_API_KEY=" >> .env
    echo "OPENAI_BASE_URL=https://openrouter.ai/api/v1" >> .env
    echo "OPENAI_MODEL=anthropic/claude-3.5-sonnet" >> .env
fi

# ========= Inject Model ID into roo-cli index.ts =========
if [ -f "src/index.ts" ]; then
    # Create backup of original file
    cp src/index.ts src/index.ts.backup
    
    # Replace the hardcoded model ID with the one passed from the evaluation
    sed -i 's|openRouterModelId: "[^"]*"|openRouterModelId: "{model}"|g' src/index.ts
    
    echo "Model ID injection complete. Verifying change..."
    grep -n "openRouterModelId" src/index.ts || echo "Warning: openRouterModelId not found in src/index.ts"
else
    echo "Warning: src/index.ts not found, model injection skipped"
fi

# ========= Run roo-cli =========
echo "Starting roo-cli with task..."
TASK_PROMPT_FROM_FILE=$(cat /tmp/task_prompt.txt)

# Change to the task directory for roo-cli execution
cd "${{WORKDIR}}"
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

dotenvx run -f /opt/roo-cli/.env -- pnpm --prefix /opt/roo-cli dev "$TASK_PROMPT_FROM_FILE" > /tmp/roo-cli-output.log 2>&1 &
PNPM_PID=$!

echo "[INFO] Waiting 5 minutes for VS Code extension to complete task..."

# Find latest Roo messages log (if present)
MSG_LOG=$(ls -1t /opt/vscode-user/User/globalStorage/rooveterinaryinc.roo-cline*/messages*.log 2>/dev/null | head -n1 || true)

# Start log tailers in background for monitoring
{{ [ -f /tmp/code.log ] && tail -F /tmp/code.log        | sed -u 's/^/[code   ] /'        & }} || true
{{ [ -f /tmp/roo-cli-output.log ] && tail -F /tmp/roo-cli-output.log | sed -u 's/^/[roo    ] /' & }} || true
{{ [ -n "$MSG_LOG" ] && tail -F "$MSG_LOG"              | sed -u 's/^/[messages] /'       & }} || true

# Clean up any previous completion signal
rm -f /tmp/roo-finish.log

# Wait for completion signal or timeout after 10 minutes
echo "[INFO] Waiting for task completion signal..."
TIMEOUT_SECONDS=240 # 4 minutes max
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    
    # Check if completion file exists
    if [ -f /tmp/roo-finish.log ]; then
        echo "[COMPLETION] Task completion signal received after ${{ELAPSED_TIME}} seconds"
        echo "[COMPLETION] Contents: $(cat /tmp/roo-finish.log 2>/dev/null || echo 'empty')"
        break
    fi
    
    # Check timeout
    if [ $ELAPSED_TIME -ge $TIMEOUT_SECONDS ]; then
        echo "[TIMEOUT] No completion signal after ${{TIMEOUT_SECONDS}} seconds - proceeding anyway"
        break
    fi
    
    # Status update every 30 seconds
    if [ $((ELAPSED_TIME % 30)) -eq 0 ] && [ $ELAPSED_TIME -gt 0 ]; then
        echo "[STATUS] Waiting for completion signal... (${{ELAPSED_TIME}}s elapsed)"
    fi
    
    sleep 5
done
"""


# =============================================================================
# Output Formatting
# =============================================================================


def format_solver_output(
    code_agent: str, setup_out: str, code_agent_out: str, test_out: str
) -> str:
    """Format the final solver output consistently across code agents.

    Args:
        code_agent: The CLI code agent that was used
        setup_out: Output from setup commands
        code_agent_out: Output from the CLI code agent execution
        test_out: Output from final test execution

    Returns:
        Formatted completion string
    """
    code_agent_section_map = {
        "aider": "AIDER_OUTPUT",
        "opencode": "OPENCODE_OUTPUT",
        "claude": "CLAUDE_CODE_OUTPUT",
        "roo": "ROO_CLI_EXECUTION",
    }

    code_agent_section = code_agent_section_map.get(
        code_agent, f"{code_agent.upper()}_OUTPUT"
    )

    return "\n".join(
        [
            "[SETUP_OUTPUT]",
            setup_out,
            "",
            f"[{code_agent_section}]",
            code_agent_out,
            "",
            "[FINAL_TEST_RESULTS]",
            test_out,
        ]
    )
