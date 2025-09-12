"""
Utility functions for CLI-based solvers that run tasks inside Docker sandboxes.

This module provides common functionality for different CLI harnesses (aider, opencode, roo)
including repository management, environment setup, and command execution.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from inspect_ai.util import sandbox


# =============================================================================
# Common Repository and Environment Management
# =============================================================================


async def ensure_repo_and_task(language: str, task_name: str) -> bool:
    """Clone Roo-Code-Evals into /workspace if needed and verify task exists.

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
        result = await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} && {test_command}"],
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
    return [
        # Core providers
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        # OpenBench supported providers
        "AI21_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "BASETEN_API_KEY",
        "CEREBRAS_API_KEY",
        "COHERE_API_KEY",
        "CRUSOE_API_KEY",
        "DEEPINFRA_API_KEY",
        "FRIENDLI_TOKEN",  # Note: Friendli uses TOKEN not API_KEY
        "HF_TOKEN",  # Hugging Face
        "HYPERBOLIC_API_KEY",
        "LAMBDA_API_KEY",
        "MINIMAX_API_KEY",
        "MOONSHOT_API_KEY",
        "NEBIUS_API_KEY",
        "NOUS_API_KEY",
        "NOVITA_API_KEY",
        "PARASAIL_API_KEY",
        "REKA_API_KEY",
        "SAMBANOVA_API_KEY",
        "AI_GATEWAY_API_KEY",  # Vercel
        # Other common providers
        "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "DEEPSEEK_API_KEY",
        "PERPLEXITY_API_KEY",
        "XAI_API_KEY",
        # AWS credentials
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]


def collect_provider_env() -> Dict[str, str]:
    """Collect a comprehensive set of provider API keys from host environment.

    Defaults to empty strings when not present, so the sandbox always receives
    a consistent mapping without leaking missing envs as None.

    Returns:
        Dictionary mapping environment variable names to their values
    """
    keys = get_provider_env_keys()
    env: Dict[str, str] = {}
    for key in keys:
        env[key] = os.getenv(key, "")
    return env


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


# =============================================================================
# Harness-Specific Command Builders
# =============================================================================


async def build_aider_command(workdir: str, prompt_text: str, model: str) -> str:
    """Build and execute aider CLI command with comprehensive environment setup.

    Args:
        workdir: Working directory path for the task
        prompt_text: The prompt to send to aider
        model: Model string to use with aider

    Returns:
        Formatted output string with aider execution results
    """
    try:
        # Write prompt to a temp file to avoid shell quoting issues
        write_prompt = await sandbox().exec(
            cmd=[
                "bash",
                "-lc",
                "cat > /tmp/aider_prompt.txt <<'EOF'\n" + prompt_text + "\nEOF",
            ],
            timeout=15,
        )
        if write_prompt.returncode != 0:
            return f"ERROR: failed to write prompt: {write_prompt.stderr}"

        # Initialize git repository if not already present (Aider requires git)
        await sandbox().exec(
            cmd=["bash", "-lc", f"cd {workdir} || true"],
            timeout=60,
        )

        # Get environment setup script
        env_setup = generate_env_setup_script()

        # Create aider execution script
        script_content = f"""#!/bin/bash
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

        # Write the script
        script_write = await sandbox().exec(
            cmd=[
                "bash",
                "-c",
                f"cat > /tmp/aider_script.sh <<'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF",
            ],
            timeout=30,
        )
        if script_write.returncode != 0:
            return f"ERROR: failed to write aider script: {script_write.stderr}"

        # Make script executable
        chmod_result = await sandbox().exec(
            cmd=["chmod", "+x", "/tmp/aider_script.sh"],
            timeout=30,
        )
        if chmod_result.returncode != 0:
            return f"ERROR: failed to make script executable: {chmod_result.stderr}"

        # Execute the aider script
        result = await sandbox().exec(
            cmd=["/tmp/aider_script.sh"],
            timeout=1800,  # 30 minutes timeout
            env=collect_provider_env(),
        )

        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])

        # Append the captured aider log if present
        try:
            log_read = await sandbox().exec(
                cmd=[
                    "bash",
                    "-lc",
                    "cat /tmp/aider-output.log || echo 'No aider output log found'",
                ],
                timeout=10,
            )
            if log_read.stdout:
                parts.extend(["", "--- AIDER LOG ---", log_read.stdout])
        except Exception:
            parts.append("--- AIDER LOG ---\nFailed to read aider output log")

        return "\n".join(parts)

    except Exception as e:
        return f"ERROR: Failed to run aider: {str(e)}"


async def build_opencode_command(workdir: str, prompt_text: str, model: str) -> str:
    """Build and execute opencode CLI command.

    Args:
        workdir: Working directory path for the task
        prompt_text: The prompt to send to opencode
        model: Model string to use with opencode

    Returns:
        Formatted output string with opencode execution results
    """
    try:
        # Write prompt to a temp file to avoid shell quoting issues
        write_prompt = await sandbox().exec(
            cmd=[
                "bash",
                "-lc",
                "cat > /tmp/opencode_prompt.txt <<'EOF'\n" + prompt_text + "\nEOF",
            ],
            timeout=15,
        )
        if write_prompt.returncode != 0:
            return f"ERROR: failed to write prompt: {write_prompt.stderr}"

        # Run opencode directly
        command = (
            f"cd /workspace && cd {workdir} && "
            f"PROMPT=$(cat /tmp/opencode_prompt.txt) && "
            f'opencode run -m {model} "$PROMPT" 2>&1 | tee /tmp/opencode-output.log'
        )

        result = await sandbox().exec(
            cmd=["bash", "-lc", command],
            timeout=1800,
            env=collect_provider_env(),
        )

        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])

        # Append the captured log if present
        try:
            log_read = await sandbox().exec(
                cmd=["bash", "-lc", "tail -n 200 /tmp/opencode-output.log || true"],
                timeout=10,
            )
            if log_read.stdout:
                parts.extend(["", "--- OPENCODE LOG (tail) ---", log_read.stdout])
        except Exception:
            pass

        return "\n".join(parts)

    except Exception as e:
        return f"ERROR: Failed to run opencode: {str(e)}"


async def build_claude_code_command(workdir: str, prompt_text: str, model: str) -> str:
    """Build and execute claude code CLI command.

    Args:
        workdir: Working directory path for the task
        prompt_text: The prompt to send to claude code
        model: Model string to use with claude code

    Returns:
        Formatted output string with claude code execution results
    """
    try:
        # Write prompt to a temp file to avoid shell quoting issues
        write_prompt = await sandbox().exec(
            cmd=[
                "bash",
                "-lc",
                "cat > /tmp/claude_code_prompt.txt <<'EOF'\n" + prompt_text + "\nEOF",
            ],
            timeout=15,
        )
        if write_prompt.returncode != 0:
            return f"ERROR: failed to write prompt: {write_prompt.stderr}"

        # Check for ANTHROPIC_API_KEY
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            return "ERROR: ANTHROPIC_API_KEY is not set"

        command = (
            f'export ANTHROPIC_API_KEY="{anthropic_api_key}" && '
            f"cd /workspace && cd {workdir} && "
            f'cat /tmp/claude_code_prompt.txt | claude -p --model "{model}" '
            f"--permission-mode acceptEdits "
            f'--allowedTools "Bash(*)" "Read" "Edit" '
            f"2>&1 | tee /tmp/claude-code-output.log"
        )

        result = await sandbox().exec(
            cmd=["bash", "-lc", command],
            timeout=1800,
            env=collect_provider_env(),
        )

        parts: List[str] = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]
        if result.stdout:
            parts.extend(["", "--- STDOUT ---", result.stdout])
        if result.stderr:
            parts.extend(["", "--- STDERR ---", result.stderr])

        # Append the captured log if present
        try:
            log_read = await sandbox().exec(
                cmd=["bash", "-lc", "tail -n 200 /tmp/claude-code-output.log || true"],
                timeout=10,
            )
            if log_read.stdout:
                parts.extend(["", "--- CLAUDE CODE LOG (tail) ---", log_read.stdout])
        except Exception:
            pass

        return "\n".join(parts)

    except Exception as e:
        return f"ERROR: Failed to run claude code: {str(e)}"


async def build_roo_command(
    workdir: str, prompt_text: str, model: Optional[str] = None
) -> str:
    """Build and execute roo CLI command with VS Code headless mode.

    Args:
        workdir: Working directory path for the task
        prompt_text: The prompt to send to roo-cli
        model: Model string (ignored for roo, uses fixed OpenRouter config)

    Returns:
        Formatted output string with roo execution results
    """
    # Get environment setup script
    env_setup = generate_env_setup_script()

    # Add completion instruction to the prompt
    enhanced_prompt = f"""{prompt_text}

IMPORTANT: When you have completed the task, please run this exact command to signal completion:
echo "Task completed at $(date)" > /tmp/roo-finish.log

This will allow the evaluation system to know when you're done."""

    # Create the roo-cli execution script content
    script_content = f"""#!/bin/bash
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

# ========= Run roo-cli =========
echo "Starting roo-cli with task..."
TASK_PROMPT_FROM_FILE=$(cat /tmp/task_prompt.txt)

# Change to the task directory for roo-cli execution
cd "${{WORKDIR}}"
echo "Current working directory: $(pwd)"
echo "Directory contents:"
ls -la

dotenvx run -f /opt/roo-cli/.env -- pnpm --prefix /opt/roo-cli dev --model {model} "$TASK_PROMPT_FROM_FILE" > /tmp/roo-cli-output.log 2>&1 &
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

    try:
        # First, write the script to a file in the container
        script_write_result = await sandbox().exec(
            cmd=[
                "bash",
                "-c",
                f"cat > /tmp/roo_cli_script.sh << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF",
            ],
            timeout=30,
        )

        if script_write_result.returncode != 0:
            return f"ERROR: Failed to write script file: {script_write_result.stderr}"

        # Make the script executable
        chmod_result = await sandbox().exec(
            cmd=["chmod", "+x", "/tmp/roo_cli_script.sh"],
            timeout=30,
        )

        if chmod_result.returncode != 0:
            return f"ERROR: Failed to make script executable: {chmod_result.stderr}"

        # Execute the script
        result = await sandbox().exec(
            cmd=["/tmp/roo_cli_script.sh"],
            timeout=1800,
            env={
                "WORKDIR": workdir,
                "TASK_PROMPT": prompt_text,
                "ROO_CODE_IPC_SOCKET_PATH": "/tmp/roo-code.sock",
                "VSCODE_EXT_DIR": "/opt/vscode-extensions",
                "VSCODE_USER_DIR": "/opt/vscode-user",
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            },
        )

        output_parts = [
            f"Exit Code: {result.returncode}",
            f"Success: {result.returncode == 0}",
        ]

        if result.stdout:
            output_parts.extend(["", "--- STDOUT ---", result.stdout])

        if result.stderr:
            output_parts.extend(["", "--- STDERR ---", result.stderr])

        # Capture roo-cli specific logs
        try:
            roo_output_result = await sandbox().exec(
                cmd=[
                    "bash",
                    "-c",
                    "cat /tmp/roo-cli-output.log || echo 'No roo-cli output found'",
                ],
                timeout=10,
            )
            if roo_output_result.stdout:
                output_parts.extend(
                    ["", "--- ROO-CLI OUTPUT ---", roo_output_result.stdout]
                )
        except Exception:
            output_parts.append("--- ROO-CLI OUTPUT ---\nFailed to read roo-cli output")

        # Capture VS Code logs
        try:
            vscode_log_result = await sandbox().exec(
                cmd=[
                    "bash",
                    "-c",
                    "cat /tmp/code.log || echo 'No VS Code log found'",
                ],
                timeout=10,
            )
            if vscode_log_result.stdout:
                output_parts.extend(
                    ["", "--- VSCODE LOG ---", vscode_log_result.stdout]
                )
        except Exception:
            output_parts.append("--- VSCODE LOG ---\nFailed to read VS Code log")

        return "\n".join(output_parts)

    except Exception as e:
        return f"ERROR: Failed to run roo-cli: {str(e)}"


# =============================================================================
# Model Parameter Handling
# =============================================================================


def resolve_model_for_harness(harness: str, state_model: str, task_args: Dict) -> str:
    """Resolve the appropriate model string based on harness requirements.

    Args:
        harness: The CLI harness being used ('aider', 'opencode', 'claude', 'roo')
        state_model: Model from TaskState.model
        task_args: Task arguments that may contain overrides

    Returns:
        Resolved model string for the harness
    """
    if harness == "aider":
        # Aider uses model directly from state
        return str(state_model) if state_model else "openai/gpt-4o-mini"

    elif harness == "opencode":
        # OpenCode supports environment override with fallback
        selected_model = os.getenv("OPEN_CODE_MODEL")
        if not selected_model:
            selected_model = str(state_model) if state_model else None
        return selected_model or "openai/gpt-4o-mini"

    elif harness == "claude":
        # Claude CLI uses Anthropic models directly
        if state_model:
            model_str = str(state_model)
            # Remove anthropic/ prefix if present
            if model_str.startswith("anthropic/"):
                return model_str[len("anthropic/") :]
            return model_str
        else:
            return "claude-sonnet-4-20250514"

    elif harness == "roo":
        # Roo uses fixed OpenRouter configuration
        if state_model:
            return str(state_model)
        else:
            return "anthropic/claude-sonnet-4-20250514"  # Fixed for roo harness

    else:
        # Unknown harness, default to state model
        return str(state_model) if state_model else "openai/gpt-4o-mini"


# =============================================================================
# Output Formatting
# =============================================================================


def format_solver_output(
    harness: str, setup_out: str, harness_out: str, test_out: str
) -> str:
    """Format the final solver output consistently across harnesses.

    Args:
        harness: The CLI harness that was used
        setup_out: Output from setup commands
        harness_out: Output from the CLI harness execution
        test_out: Output from final test execution

    Returns:
        Formatted completion string
    """
    harness_section_map = {
        "aider": "AIDER_OUTPUT",
        "opencode": "OPENCODE_OUTPUT",
        "claude": "CLAUDE_CODE_OUTPUT",
        "roo": "ROO_CLI_EXECUTION",
    }

    harness_section = harness_section_map.get(harness, f"{harness.upper()}_OUTPUT")

    return "\n".join(
        [
            "[SETUP_OUTPUT]",
            setup_out,
            "",
            f"[{harness_section}]",
            harness_out,
            "",
            "[FINAL_TEST_RESULTS]",
            test_out,
        ]
    )
