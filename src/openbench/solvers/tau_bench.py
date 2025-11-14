"""
Async solver that runs tau2-bench simulations against Inspect models.

This leverages certain parts of the tau2 package (like the construction of the env, the prompts, the execution etc.),
but uses inspect to call the LLMs (to leverage their integration and to use the API keys of the user).
This, however, means that the state has to be passed constantly between inspect and tau2-bench.

As it re-implements parts of tau2, parts are adapted from the original implementation,
see https://github.com/sierra-research/tau2-bench (MIT)
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Model,
    get_model,
)
from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolError
from inspect_ai.tool import ToolCall as InspectToolCall
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams
from inspect_ai.model import GenerateConfig
from rich.console import Console
from inspect_ai._display.textual.display import TextualDisplay  # type: ignore

_tau2_logging_configured = False


def _ensure_tau2_logging_redirected() -> None:
    """Route tau2's loguru output through Inspect's display console."""

    global _tau2_logging_configured
    if _tau2_logging_configured:
        return

    try:
        from loguru import logger as loguru_logger  # type: ignore
    except ImportError:
        _tau2_logging_configured = True
        return

    fallback_console = Console()

    def _console_sink(message: str) -> None:
        text = message.rstrip("\n")
        try:
            from inspect_ai._display.core.active import display as get_display

            disp = get_display()
            ctx = (
                nullcontext()
                if isinstance(disp, TextualDisplay)
                else disp.suspend_task_app()
            )
            with ctx:
                disp.print(text)
        except Exception:
            fallback_console.print(text, markup=False, highlight=False)

    loguru_logger.remove()
    loguru_logger.add(
        _console_sink,
        level="INFO",
        enqueue=False,  # keep intra-process queueing simple (no mp lock needed)
        format="[tau2] {time:HH:mm:ss} | {level:<8} | {message}",
    )
    _tau2_logging_configured = True


@lru_cache(maxsize=1)
def _tau2() -> SimpleNamespace:
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # type: ignore
    from tau2.data_model.message import (  # type: ignore
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.data_model.simulation import SimulationRun, TerminationReason  # type: ignore
    from tau2.data_model.tasks import Task  # type: ignore
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # type: ignore
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Role  # type: ignore
    from tau2.user.base import OUT_OF_SCOPE, STOP, TRANSFER  # type: ignore
    from tau2.user.user_simulator import get_global_user_sim_guidelines  # type: ignore
    from tau2.utils.utils import get_now  # type: ignore

    return SimpleNamespace(  # type: ignore
        AGENT_INSTRUCTION=AGENT_INSTRUCTION,
        SYSTEM_PROMPT=SYSTEM_PROMPT,
        AssistantMessage=AssistantMessage,
        UserMessage=UserMessage,
        ToolMessage=ToolMessage,
        ToolCall=ToolCall,
        SimulationRun=SimulationRun,
        TerminationReason=TerminationReason,
        Task=Task,
        EvaluationType=EvaluationType,
        evaluate_simulation=evaluate_simulation,
        DEFAULT_FIRST_AGENT_MESSAGE=DEFAULT_FIRST_AGENT_MESSAGE,
        Role=Role,
        OUT_OF_SCOPE=OUT_OF_SCOPE,
        STOP=STOP,
        TRANSFER=TRANSFER,
        get_global_user_sim_guidelines=get_global_user_sim_guidelines,
        get_now=get_now,
    )


def _create_tau2_environment(domain: str):
    if domain == "retail":
        from tau2.domains.retail.environment import (  # type: ignore
            get_environment as get_retail_env,
        )

        return get_retail_env()
    if domain == "airline":
        from tau2.domains.airline.environment import (  # type: ignore
            get_environment as get_airline_env,
        )

        return get_airline_env()
    if domain == "telecom":
        from tau2.domains.telecom.environment import (  # type: ignore
            get_environment as get_telecom_env,
        )

        return get_telecom_env()
    if domain == "mock":
        from tau2.domains.mock.environment import (  # type: ignore
            get_environment as get_mock_env,
        )

        return get_mock_env()
    raise ValueError(f"Unsupported tau-bench domain: {domain}")


def _user_system_prompt(use_tools: bool, instructions: str) -> str:
    """
    In tau2, this is the system_prompt property of UserSimulator(BaseUser)
    """
    guidelines = _tau2().get_global_user_sim_guidelines(use_tools=use_tools)
    template = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()
    return template.format(
        global_user_sim_guidelines=guidelines,
        instructions=instructions,
    )


def _convert_tools(tools) -> list[Tool]:
    if not tools:
        return []
    converted: list[Tool] = []
    for t in tools:
        schema = getattr(t, "openai_schema", None)
        if not schema:
            continue
        converted.append(_tool_from_schema(schema))
    return converted


def _tool_from_schema(schema: dict) -> Tool:
    function_meta = schema.get("function", {})
    name = function_meta.get("name") or f"tau_tool_{uuid4().hex}"
    description = function_meta.get("description", "")
    parameters_schema = function_meta.get("parameters") or {}
    if not parameters_schema:
        parameters_schema = {"type": "object", "properties": {}, "required": []}
    try:
        parameters = ToolParams.model_validate(parameters_schema)
    except Exception:
        parameters = ToolParams()

    async def _execute(**kwargs):
        raise ToolError(
            "TauBench tools are executed inside the simulator; direct execution is not supported."
        )

    _execute.__name__ = name
    tool_def = ToolDef(
        _execute,
        name=name,
        description=description,
        parameters=parameters,
        parallel=True,
    )
    return tool_def.as_tool()


def _parse_arguments(arguments: Any) -> dict:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
    return arguments


def _inspect_to_tau2_tool_call(tool_call: InspectToolCall, requestor: str):
    name = getattr(tool_call, "function", getattr(tool_call, "name", ""))
    tau2 = _tau2()
    return tau2.ToolCall(
        id=getattr(tool_call, "id", None) or f"tc_{uuid4().hex}",
        name=name,
        arguments=_parse_arguments(getattr(tool_call, "arguments", {})),
        requestor=requestor,  # type: ignore[arg-type]
    )


def _tau2_to_inspect_tool_call(tool_call) -> InspectToolCall:
    return InspectToolCall(
        id=tool_call.id or f"tc_{uuid4().hex}",
        function=tool_call.name,
        arguments=tool_call.arguments or {},
        type="function",
    )


def _tau2_to_agent_history(messages) -> list:
    chat: list = []
    tau2 = _tau2()
    for msg in messages:
        if isinstance(msg, tau2.UserMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, tau2.AssistantMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [_tau2_to_inspect_tool_call(tc) for tc in msg.tool_calls]
            chat.append(
                ChatMessageAssistant(
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, tau2.ToolMessage):
            if msg.requestor != "assistant":
                continue
            chat.append(
                ChatMessageTool(
                    content=msg.content or "",
                    tool_call_id=msg.id,
                )
            )
    return chat


def _tau2_to_user_history(messages) -> list:
    chat: list = []
    tau2 = _tau2()
    for msg in messages:
        if isinstance(msg, tau2.UserMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [_tau2_to_inspect_tool_call(tc) for tc in msg.tool_calls]
            chat.append(
                ChatMessageAssistant(
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, tau2.AssistantMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, tau2.ToolMessage):
            if msg.requestor != "user":
                continue
            chat.append(
                ChatMessageTool(
                    content=msg.content or "",
                    tool_call_id=msg.id,
                )
            )
    return chat


def _tau2_to_inspect_conversation(messages) -> list:
    chat: list = []
    tool_name_map: dict[str, str] = {}
    tau2 = _tau2()
    for msg in messages:
        if isinstance(msg, tau2.UserMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, tau2.AssistantMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    tool_calls.append(_tau2_to_inspect_tool_call(tc))
                    tool_name_map[tc.id] = tc.name
            chat.append(
                ChatMessageAssistant(
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, tau2.ToolMessage):
            chat.append(
                ChatMessageTool(
                    content=msg.content or "",
                    tool_call_id=msg.id,
                    function=tool_name_map.get(msg.id),
                )
            )
    return chat


def _is_user_stop(content: Optional[str]) -> bool:
    if not content:
        return False
    tau2 = _tau2()
    tokens = (tau2.STOP, tau2.TRANSFER, tau2.OUT_OF_SCOPE)
    return any(token in content for token in tokens)


@dataclass
class TauBenchResult:
    simulation: Any
    reward_info: Any
    termination_reason: str


class TauBenchRunner:
    """
    Minimal async reimplementation of tau2's orchestrator that swaps in Inspect models.

    Certain features (like seed, solo mode, max errors) were left out intentionally.
    """

    def __init__(
        self,
        *,
        domain: str,
        task_payload: dict,
        trial: int,
        agent_model: Model,
        user_model: Model,
        max_steps: int,
    ):
        self._tau2 = _tau2()
        self.domain = domain
        self.tau2_task = self._tau2.Task.model_validate(task_payload)
        self.trial = trial
        self.agent_model = agent_model
        self.agent_model_config = agent_model.config
        self.user_model = user_model
        self.user_model_config = user_model.config
        self.max_steps = max_steps

        self.environment = _create_tau2_environment(domain)
        self.agent_tools = _convert_tools(self.environment.get_tools())
        try:
            raw_user_tools = self.environment.get_user_tools()
        except ValueError:
            raw_user_tools = []
        self.user_tools = _convert_tools(raw_user_tools)

        instructions = str(self.tau2_task.user_scenario.instructions)
        self.agent_system_prompt = self._tau2.SYSTEM_PROMPT.format(
            agent_instruction=self._tau2.AGENT_INSTRUCTION,
            domain_policy=self.environment.get_policy(),
        )
        self.user_system_prompt = _user_system_prompt(
            use_tools=bool(self.user_tools), instructions=instructions
        )

        self.trajectory: list = []
        self.step_count = 0
        self.num_errors = 0
        self.termination_reason = None
        self.done = False

        self._initialize_environment()

    def _initialize_environment(self):
        initial_state = self.tau2_task.initial_state
        initialization_data = None
        initialization_actions = None
        message_history = []
        if initial_state:
            initialization_data = initial_state.initialization_data
            initialization_actions = initial_state.initialization_actions
            if initial_state.message_history:
                message_history = deepcopy(initial_state.message_history)

        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        for msg in message_history:
            self.trajectory.append(msg)

    async def run(self) -> TauBenchResult:
        """
        Combines tau2bench's run + step
        """
        tau2 = self._tau2
        if self.trajectory:
            last_message = self.trajectory[-1]
            if isinstance(last_message, tau2.AssistantMessage):
                to_role = (
                    tau2.Role.USER if not last_message.is_tool_call() else tau2.Role.ENV
                )
            elif isinstance(last_message, tau2.UserMessage):
                to_role = (
                    tau2.Role.AGENT
                    if not last_message.is_tool_call()
                    else tau2.Role.ENV
                )
            else:
                to_role = (
                    tau2.Role.AGENT
                    if last_message.requestor == "assistant"
                    else tau2.Role.USER
                )
        else:
            first_message = tau2.DEFAULT_FIRST_AGENT_MESSAGE
            self.trajectory.append(first_message)
            to_role = tau2.Role.USER
            last_message = first_message

        start_time = datetime.now()
        done = False

        while not done:
            if self.max_steps and self.step_count >= self.max_steps:
                self.termination_reason = tau2.TerminationReason.MAX_STEPS
                break

            if to_role == tau2.Role.USER:
                user_message = await self._generate_user_message()
                self.trajectory.append(user_message)
                if user_message.is_tool_call():
                    to_role = tau2.Role.ENV
                    last_message = user_message
                else:
                    to_role = tau2.Role.AGENT
                    last_message = user_message
                if _is_user_stop(user_message.content):
                    self.termination_reason = tau2.TerminationReason.USER_STOP
                    done = True
            elif to_role == tau2.Role.AGENT:
                agent_message = await self._generate_agent_message()
                self.trajectory.append(agent_message)
                if agent_message.is_tool_call():
                    to_role = tau2.Role.ENV
                    last_message = agent_message
                else:
                    to_role = tau2.Role.USER
                    last_message = agent_message
            else:
                tool_messages = self._execute_tools(last_message)
                self.trajectory.extend(tool_messages)
                if not tool_messages:
                    to_role = (
                        tau2.Role.USER
                        if last_message.requestor == "user"
                        else tau2.Role.AGENT
                    )
                else:
                    requestor = tool_messages[-1].requestor
                    to_role = tau2.Role.USER if requestor == "user" else tau2.Role.AGENT
                last_message = tool_messages[-1] if tool_messages else last_message
            self.step_count += 1

        end_time = datetime.now()
        final_reason = self.termination_reason or "completed"
        simulation = tau2.SimulationRun(
            id=f"{self.domain}_{self.tau2_task.id}_{uuid4().hex}",
            task_id=self.tau2_task.id,
            timestamp=tau2.get_now(),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=(end_time - start_time).total_seconds(),
            termination_reason=final_reason,  # type: ignore[arg-type]
            agent_cost=None,
            user_cost=None,
            messages=self.trajectory,
            trial=self.trial,
        )
        reward_info = tau2.evaluate_simulation(
            simulation=simulation,
            task=self.tau2_task,
            evaluation_type=tau2.EvaluationType.ALL,
            solo_mode=False,
            domain=self.domain,
        )
        return TauBenchResult(
            simulation=simulation,
            reward_info=reward_info,
            termination_reason=str(final_reason),
        )

    async def _generate_agent_message(self):
        messages = [ChatMessageSystem(content=self.agent_system_prompt)]
        messages.extend(_tau2_to_agent_history(self.trajectory))
        agent_config = self.agent_model_config or GenerateConfig(temperature=0.0)
        response = await self.agent_model.generate(
            messages,
            tools=self.agent_tools,
            config=agent_config,
            tool_choice="auto" if self.agent_tools else None,
        )
        assistant_message = getattr(response, "message", None)
        tool_calls = None
        if assistant_message and getattr(assistant_message, "tool_calls", None):
            tool_calls = [
                _inspect_to_tau2_tool_call(tc, "assistant")
                for tc in assistant_message.tool_calls
            ]
        completion = (
            assistant_message.text
            if assistant_message
            and getattr(assistant_message, "text", None) is not None
            else getattr(response, "completion", None)
        )
        return self._tau2.AssistantMessage(
            role="assistant",
            content=completion,
            tool_calls=tool_calls,
            raw_data=response.model_dump(mode="json"),
        )

    async def _generate_user_message(self):
        messages = [ChatMessageSystem(content=self.user_system_prompt)]
        messages.extend(_tau2_to_user_history(self.trajectory))
        user_config = self.user_model_config or GenerateConfig(temperature=0.0)
        response = await self.user_model.generate(
            messages,
            tools=self.user_tools,
            config=user_config,
            tool_choice="auto" if self.user_tools else None,
        )
        user_message = getattr(response, "message", None)
        tool_calls = None
        if user_message and getattr(user_message, "tool_calls", None):
            tool_calls = [
                _inspect_to_tau2_tool_call(tc, "user") for tc in user_message.tool_calls
            ]
        completion = (
            user_message.text
            if user_message and getattr(user_message, "text", None) is not None
            else getattr(response, "completion", None)
        )
        return self._tau2.UserMessage(
            role="user",
            content=completion,
            tool_calls=tool_calls,
            raw_data=response.model_dump(mode="json"),
        )

    def _execute_tools(self, message) -> list:
        tool_messages: list = []
        if not message or not message.tool_calls:
            return tool_messages
        for tc in message.tool_calls:
            response = self.environment.get_response(tc)
            tool_messages.append(response)
            if response.error:
                self.num_errors += 1
            else:
                self.num_errors = 0
        return tool_messages


@solver
def tau_bench_solver(
    *,
    user_model: str = "openai/gpt-4.1",
    max_steps: int = 200,
) -> Solver:
    """
    Solver factory for tau-bench tasks.
    """

    async def solve(state: TaskState, generate) -> TaskState:  # type: ignore[override]
        task_payload = state.metadata.get("tau2_task")
        if not task_payload:
            raise ValueError("tau2_task metadata missing from sample")
        domain = state.metadata.get("domain")
        if not domain:
            raise ValueError("domain metadata missing from sample")
        trial = state.metadata.get("trial", 1)

        candidate = get_model()
        user_model_instance = get_model(user_model)

        _ensure_tau2_logging_redirected()

        runner = TauBenchRunner(
            domain=str(domain),
            task_payload=task_payload,
            trial=trial,
            agent_model=candidate,
            user_model=user_model_instance,
            max_steps=max_steps,
        )
        result = await runner.run()
        state.metadata["tau2"] = {
            "simulation": result.simulation.model_dump(mode="json"),
            "reward_info": result.reward_info.model_dump(mode="json"),
            "termination_reason": result.termination_reason,
        }
        state.output.completion = json.dumps(
            {
                "task_id": result.simulation.task_id,
                "trial": trial,
                "reward": result.reward_info.reward,
                "termination_reason": result.termination_reason,
            },
            ensure_ascii=False,
        )
        state.messages = _tau2_to_inspect_conversation(result.simulation.messages)
        return state

    return solve
