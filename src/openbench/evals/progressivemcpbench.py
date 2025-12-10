"""
ProgressiveMCPBench evaluation.

This benchmark evaluates LLM agents on real-world tasks using MCP tools.
It supports multiple strategies for tool discovery and execution.

Strategies:
- copilot: Uses semantic search with embeddings to discover tools via route/execute-tool
- directory: Presents tools as a filesystem with ls/read-tool-file/execute-tool
- minimal-servers: Provides only the tools from the required server(s) for each task
- minimal-servers-remote: Uses Groq's server-side MCP (single-shot, groq-responses only)
- minimal-tools: Provides only the exact tools needed for each task
- distraction-64: Minimal tools plus distraction tools to total 64 tools
- distraction-128: Minimal tools plus distraction tools to total 128 tools
"""

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import ToolError, ToolSource
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from openbench.datasets.progressivemcpbench import get_dataset
from openbench.model._providers.groq_responses import GroqResponsesAPI
from openbench.scorers.progressivemcpbench import progressivemcpbench_scorer
from openbench.tools.progressivemcpbench.directory.toolsource import (
    directory_tool_source,
)
from openbench.tools.progressivemcpbench.synthetic.toolsource import (
    synthetic_minimal_servers_tool_source,
    synthetic_minimal_tools_tool_source,
    synthetic_distraction_64_tool_source,
    synthetic_distraction_128_tool_source,
)
from openbench.tools.progressivemcpbench.copilot.synthetic_toolsource import (
    synthetic_copilot_tool_source,
)
from openbench.utils.text import (
    PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE,
)
from openbench.utils.mcp_server_manager import ensure_mcp_server_running


VALID_STRATEGIES = {
    "copilot",
    "directory",
    "minimal-servers",
    "minimal-servers-remote",
    "minimal-tools",
    "distraction-64",
    "distraction-128",
}

GROQ_RESPONSES_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_PROGRESSIVE_MCP_BASE = "https://progressive-mcp-bench.groq-dev.workers.dev/mcp"
GROQ_API_KEY_ENV = "GROQ_API_KEY"


def _get_system_message(strategy: str) -> str:
    """Get the appropriate system message for the given strategy."""
    if strategy == "copilot":
        return PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE
    elif strategy == "directory":
        return PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE
    else:
        return PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE


async def _run_react_with_tools(
    state: TaskState,
    system_message: str,
    tool_source: ToolSource,
) -> TaskState:
    """Run react solver with the given tools and handle errors."""
    try:
        react_solver = react(
            prompt=AgentPrompt(
                instructions=system_message,
                assistant_prompt=None,
                handoff_prompt=None,
            ),
            tools=[tool_source],
            submit=False,
        )
        return await react_solver(state)  # type: ignore[return-value, arg-type]
    except asyncio.TimeoutError:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "timeout"
        state.metadata["error_message"] = "Task execution timed out"
        return state
    except ToolError as e:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "tool_error"
        state.metadata["error_message"] = str(e)
        if state.output and not state.output.completion:
            state.output.completion = f"Task failed due to tool error: {str(e)}"
        return state
    except Exception as e:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "runtime_error"
        state.metadata["error_message"] = str(e)
        if state.output and not state.output.completion:
            state.output.completion = f"Task failed due to runtime error: {str(e)}"
        return state


@solver
def progressive_copilot_solver() -> Solver:
    """Solver that uses the synthetic Copilot MCP server for ProgressiveMCPBench.

    This strategy uses semantic search with embeddings to discover tools,
    then executes them via the synthetic HTTP MCP backend.
    """

    async def solve(state: TaskState, generate: Any) -> TaskState:
        tool_source = synthetic_copilot_tool_source()
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_directory_solver() -> Solver:
    """Solver that uses the Directory MCP server for ProgressiveMCPBench."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        tool_source = directory_tool_source()
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_minimal_servers_solver() -> Solver:
    """Solver that provides all tools from the required server(s) for each task."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])

        if not required_servers:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' annotation."
            )
            return state

        tool_source = synthetic_minimal_servers_tool_source(required_servers)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_minimal_tools_solver() -> Solver:
    """Solver that provides only the exact tools needed for each task."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_minimal_tools_tool_source(required_tools)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


def _build_required_tools(
    required_servers: list[str], required_tools_list: list[str]
) -> list[tuple[str, str]]:
    """Build list of (server, tool) tuples from annotations."""
    required_tools: list[tuple[str, str]] = []
    for server in required_servers:
        for tool in required_tools_list:
            required_tools.append((server, tool))
    return required_tools


@solver
def progressive_distraction_64_solver() -> Solver:
    """Solver that provides required tools plus distractors to total 64 tools."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])
        task_id = str(state.sample_id or "")

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_distraction_64_tool_source(required_tools, task_id)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_distraction_128_solver() -> Solver:
    """Solver that provides required tools plus distractors to total 128 tools."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])
        task_id = str(state.sample_id or "")

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_distraction_128_tool_source(required_tools, task_id)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


def _get_synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _load_servers_config() -> dict[str, Any]:
    """Load the synthetic servers.json configuration."""
    config_path = _get_synthetic_mcp_dir() / "config" / "servers.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@solver
def progressive_minimal_servers_remote_solver() -> Solver:
    """Solver for minimal-servers-remote strategy using Groq's server-side MCP.

    This strategy:
    - Makes a single call to Groq Responses API with MCP server specs
    - Groq handles all tool discovery and execution internally
    - No local tool-calling loop (unlike other strategies)
    - Only works with groq-responses provider

    The remote MCP server is at: https://progressive-mcp-bench.groq-dev.workers.dev/mcp/<server>
    """

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])

        if not required_servers:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' annotation."
            )
            return state

        model = get_model()

        if not isinstance(model.api, GroqResponsesAPI):
            raise RuntimeError(
                f"The 'minimal-servers-remote' strategy only works with the "
                f"'groq-responses' provider. Got model: {model.name}. "
                f"Use a model like: groq-responses/openai/gpt-oss-20b"
            )

        groq_model_id = model.name

        servers_config = _load_servers_config()

        mcp_tools = []
        for server_name in required_servers:
            server_desc = ""
            if server_name in servers_config:
                server_desc = servers_config[server_name].get("description", "")
            if not server_desc:
                server_desc = f"MCP server '{server_name}' for ProgressiveMCPBench"

            mcp_tools.append(
                {
                    "type": "mcp",
                    "server_label": server_name,
                    "server_url": f"{GROQ_PROGRESSIVE_MCP_BASE}/{server_name}",
                    "require_approval": "never",
                    "server_description": server_desc,
                }
            )

        system_message = _get_system_message("minimal-servers")
        user_text = state.input_text

        groq_input = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
        ]

        api_key = os.environ.get(GROQ_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"GROQ API key not found in environment variable {GROQ_API_KEY_ENV}. "
                f"Required for 'minimal-servers-remote' strategy."
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=GROQ_RESPONSES_BASE_URL,
        )

        request_payload = {
            "model": groq_model_id,
            "input": groq_input,
            "tools": mcp_tools,
            "stream": False,
            "max_output_tokens": 2048,
            "temperature": 0.7,
        }

        try:
            response = await client.responses.create(
                model=groq_model_id,
                input=groq_input,  # type: ignore[arg-type]
                tools=mcp_tools,  # type: ignore[arg-type]
                stream=False,
                max_output_tokens=2048,
                temperature=0.7,
            )
        except Exception as e:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "api_error"
            state.metadata["error_message"] = str(e)
            state.metadata["groq_mcp_request"] = request_payload
            if state.output:
                state.output.completion = f"API error: {str(e)}"
            return state
        finally:
            await client.close()

        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}

        output_items = getattr(response, "output", []) or []

        assistant_text_chunks: list[str] = []
        tool_call_detected = False
        mcp_events: list[dict[str, Any]] = []

        for item in output_items:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        assistant_text_chunks.append(getattr(c, "text", ""))
            elif item_type == "function_call":
                tool_call_detected = True
            elif item_type.startswith("mcp_"):
                mcp_events.append(
                    item.model_dump()
                    if hasattr(item, "model_dump")
                    else {"type": item_type}
                )

        assistant_text = "".join(assistant_text_chunks).strip()

        state.metadata = state.metadata or {}
        state.metadata["groq_mcp_request"] = request_payload
        state.metadata["groq_mcp_response"] = response_dict
        state.metadata["groq_mcp_events"] = mcp_events

        if tool_call_detected:
            state.metadata["execution_error"] = "tool_calls_not_allowed"
            state.metadata["error_message"] = (
                "Groq Responses produced tool calls for 'minimal-servers-remote'. "
                "This strategy requires single-shot completion - the remote MCP "
                "server should handle all tool execution internally."
            )

        if state.output:
            state.output.completion = assistant_text
        else:
            state.output = type("Output", (), {"completion": assistant_text})()

        return state

    return solve


def _get_solver_for_strategy(strategy: str) -> Solver:
    """Get the appropriate solver for the given strategy."""
    if strategy == "copilot":
        return progressive_copilot_solver()
    elif strategy == "directory":
        return progressive_directory_solver()
    elif strategy == "minimal-servers":
        return progressive_minimal_servers_solver()
    elif strategy == "minimal-servers-remote":
        return progressive_minimal_servers_remote_solver()
    elif strategy == "minimal-tools":
        return progressive_minimal_tools_solver()
    elif strategy == "distraction-64":
        return progressive_distraction_64_solver()
    elif strategy == "distraction-128":
        return progressive_distraction_128_solver()
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Valid strategies: {VALID_STRATEGIES}"
        )


@task
def progressivemcpbench(
    working_limit: int = 60,
    strategy: str | None = None,
) -> Task:
    """ProgressiveMCPBench using configurable tool discovery strategies.

    All strategies now use the synthetic MCP layer with deterministic responses.

    Args:
        working_limit: Maximum number of API calls per task.
        strategy: Tool discovery strategy. Required. One of:
            - "copilot": Semantic search with embeddings (route/execute-tool)
            - "directory": Filesystem-like exploration (ls/read-tool-file/execute-tool)
            - "minimal-servers": Direct access to required server tools (requires annotations)
            - "minimal-servers-remote": Groq server-side MCP (single-shot, groq-responses only)
            - "minimal-tools": Direct access to exact required tools (requires annotations)
            - "distraction-64": Minimal tools + distractors to 64 total (requires annotations)
            - "distraction-128": Minimal tools + distractors to 128 total (requires annotations)
    """
    if strategy is None:
        raise ValueError(
            "The 'strategy' task variable is required for ProgressiveMCPBench.\n"
            "Use -T strategy=directory or -T strategy=minimal-tools on the command line.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    ensure_mcp_server_running()

    solver = _get_solver_for_strategy(strategy)

    # All strategies now use the synthetic dataset
    dataset = get_dataset()

    return Task(
        dataset=dataset,
        solver=[solver],
        scorer=progressivemcpbench_scorer(),
        name=f"progressivemcpbench-{strategy}",
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
            # extra_body={"response_format": {"type": "json_object"}}, # removed to avoid 'json' tool issues
        ),
        working_limit=working_limit,
    )
