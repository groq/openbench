from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import ToolError
import asyncio

from openbench.datasets.progressivemcpbench import get_dataset
from openbench.scorers.progressivemcpbench import progressivemcpbench_scorer
from openbench.tools.progressivemcpbench.copilot.toolsource import copilot_tool_source
from openbench.tools.progressivemcpbench.directory.toolsource import (
    directory_tool_source,
)
from openbench.tools.progressivemcpbench.direct import (
    DirectToolSourceFactory,
    parse_required_servers,
    parse_required_tools,
)
from openbench.utils.text import progressivemcpbench_system_message


SUPPORTED_STRATEGIES = {
    "copilot",
    "directory",
    "minimal-servers",
    "minimal-tasks",
    "distraction-128",
}


def _resolve_strategy(strategy: str | None) -> str:
    normalized = (strategy or "").strip().lower()
    if not normalized:
        raise ValueError(
            "ProgressiveMCPBench requires a strategy (-T strategy=copilot|directory)."
        )
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Choose one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
        )
    return normalized


def _tool_source_builder(strategy: str):
    if strategy == "copilot":
        ts = copilot_tool_source()
        return lambda _state: [ts]
    if strategy == "directory":
        ts = directory_tool_source()
        return lambda _state: [ts]
    if strategy in {"minimal-servers", "minimal-tasks", "distraction-128"}:
        factory = DirectToolSourceFactory()

        def build(state):
            metadata = state.metadata or {}
            sample_id = str(state.sample_id)
            if strategy == "minimal-servers":
                required_servers = parse_required_servers(metadata)
                required_servers = list(dict.fromkeys(required_servers))
                if not required_servers:
                    raise ValueError(
                        f"Task {sample_id} missing required_servers metadata for minimal-servers strategy."
                    )
                factory.require_servers(required_servers)
                tool_sources = []
                for server in required_servers:
                    tools = factory.server_tools(server)
                    if not tools:
                        raise ValueError(
                            f"Server '{server}' has no tools configured for task {sample_id}."
                        )
                    tool_sources.append(factory.build_tool_source(server, tools))
                return tool_sources

            if strategy == "minimal-tasks":
                required_tools = parse_required_tools(metadata)
                required_tools = list(dict.fromkeys(required_tools))
                if not required_tools:
                    raise ValueError(
                        f"Task {sample_id} missing required_tools metadata for minimal-tasks strategy."
                    )
                factory.require_tools(required_tools)
                grouped: dict[str, list[str]] = {}
                for server, tool in required_tools:
                    grouped.setdefault(server, []).append(tool)
                return [
                    factory.build_tool_source(server, list(dict.fromkeys(tools)))
                    for server, tools in grouped.items()
                ]

            if strategy == "distraction-128":
                required_tools = parse_required_tools(metadata)
                required_servers = parse_required_servers(metadata)
                required_tools = list(dict.fromkeys(required_tools))
                required_servers = list(dict.fromkeys(required_servers))

                base_tools: list[tuple[str, str]] = []
                if required_tools:
                    base_tools = required_tools
                elif required_servers:
                    factory.require_servers(required_servers)
                    for server in required_servers:
                        for tool in factory.server_tools(server):
                            base_tools.append((server, tool))
                base_tools = list(dict.fromkeys(base_tools))

                if not base_tools:
                    raise ValueError(
                        f"Task {sample_id} missing required tool metadata for distraction-128 strategy."
                    )

                factory.require_tools(base_tools)
                fillers = factory.deterministic_distractors(
                    task_id=sample_id,
                    required=base_tools,
                    target_total=128,
                )
                combined_tools = base_tools + fillers
                grouped: dict[str, list[str]] = {}
                for server, tool in combined_tools:
                    grouped.setdefault(server, []).append(tool)
                return [
                    factory.build_tool_source(server, list(dict.fromkeys(tools)))
                    for server, tools in grouped.items()
                ]

            raise ValueError(f"Unsupported strategy '{strategy}'")

        return build

    raise ValueError(f"Unsupported strategy '{strategy}'")


@solver
def progressive_solver(strategy: str) -> Solver:
    """Solver that routes to a specific ProgressiveMCPBench strategy."""
    resolved_strategy = _resolve_strategy(strategy)
    instructions = progressivemcpbench_system_message(resolved_strategy)
    tool_source_builder = _tool_source_builder(resolved_strategy)

    async def solve(state: TaskState, generate) -> TaskState:
        try:
            tool_sources = tool_source_builder(state)
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=instructions,
                    assistant_prompt=None,
                    handoff_prompt=None,
                ),
                tools=tool_sources,
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

    return solve


@task
def progressivemcpbench(
    strategy: str,
    working_limit: int = 600,
) -> Task:
    """ProgressiveMCPBench with selectable tool discovery strategies."""

    resolved_strategy = _resolve_strategy(strategy)

    return Task(
        dataset=get_dataset(),
        solver=[progressive_solver(resolved_strategy)],
        scorer=progressivemcpbench_scorer(),
        name="progressivemcpbench",
        metadata={"strategy": resolved_strategy},
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
            extra_body={"response_format": {"type": "json_object"}},
        ),
        working_limit=working_limit,
    )
