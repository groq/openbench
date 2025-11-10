"""
Async solver that runs tau2-bench simulations against Inspect models.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional
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


def _load_tau2_task(task_payload: dict):
    from tau2.data_model.tasks import Task as Tau2Task  # type: ignore

    return Tau2Task.model_validate(task_payload)


def _tau2_now() -> str:
    from tau2.utils.utils import get_now  # type: ignore

    return get_now()


def _default_first_agent_message():
    from tau2.orchestrator.orchestrator import (  # type: ignore
        DEFAULT_FIRST_AGENT_MESSAGE,
    )

    first = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
    first.timestamp = _tau2_now()
    return first


def _agent_system_prompt(policy: str) -> str:
    from tau2.agent.llm_agent import (  # type: ignore
        AGENT_INSTRUCTION,
        SYSTEM_PROMPT,
    )

    return SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION, domain_policy=policy
    )


def _user_system_prompt(use_tools: bool, instructions: str) -> str:
    from tau2.user.user_simulator import (  # type: ignore
        get_global_user_sim_guidelines,
    )

    guidelines = get_global_user_sim_guidelines(use_tools=use_tools)
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


def _user_stop_tokens() -> Iterable[str]:
    from tau2.user.base import (  # type: ignore
        OUT_OF_SCOPE,
        STOP,
        TRANSFER,
    )

    return (STOP, TRANSFER, OUT_OF_SCOPE)


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
    tdef = ToolDef(
        _execute,
        name=name,
        description=description,
        parameters=parameters,
        parallel=True,
    )
    return tdef.as_tool()


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
    from tau2.data_model.message import ToolCall  # type: ignore

    name = getattr(tool_call, "function", getattr(tool_call, "name", ""))
    return ToolCall(
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
    for msg in messages:
        from tau2.data_model.message import (  # type: ignore
            AssistantMessage,
            ToolMessage,
            UserMessage,
        )

        if isinstance(msg, UserMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, AssistantMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [_tau2_to_inspect_tool_call(tc) for tc in msg.tool_calls]
            chat.append(
                ChatMessageAssistant(
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, ToolMessage):
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
    for msg in messages:
        from tau2.data_model.message import (  # type: ignore
            AssistantMessage,
            ToolMessage,
            UserMessage,
        )

        if isinstance(msg, UserMessage):
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [_tau2_to_inspect_tool_call(tc) for tc in msg.tool_calls]
            chat.append(
                ChatMessageAssistant(
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, AssistantMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, ToolMessage):
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
    from tau2.data_model.message import (  # type: ignore
        AssistantMessage,
        ToolMessage,
        UserMessage,
    )

    chat: list = []
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, UserMessage):
            chat.append(ChatMessageUser(content=msg.content or ""))
        elif isinstance(msg, AssistantMessage):
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
        elif isinstance(msg, ToolMessage):
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
    tokens = _user_stop_tokens()
    return any(token in content for token in tokens)


@dataclass
class TauBenchResult:
    simulation: Any
    reward_info: Any
    termination_reason: str


class TauBenchRunner:
    """
    Minimal async reimplementation of tau2's orchestrator that swaps in Inspect models.
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
        max_errors: int,
    ):
        self.domain = domain
        self.tau2_task = _load_tau2_task(task_payload)
        self.trial = trial
        self.agent_model = agent_model
        self.user_model = user_model
        self.max_steps = max_steps
        self.max_errors = max_errors

        self.environment = _create_tau2_environment(domain)
        self.agent_tools = _convert_tools(self.environment.get_tools())
        try:
            self.user_tools = _convert_tools(self.environment.get_user_tools())
        except Exception:
            self.user_tools = []

        instructions = str(self.tau2_task.user_scenario.instructions)
        self.agent_system_prompt = _agent_system_prompt(self.environment.get_policy())
        self.user_system_prompt = _user_system_prompt(
            use_tools=bool(self.user_tools), instructions=instructions
        )

        self.trajectory: list = []
        self.step_count = 0
        self.num_errors = 0
        self.termination_reason = None

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
        from tau2.data_model.message import (  # type: ignore
            AssistantMessage,
            UserMessage,
        )
        from tau2.data_model.simulation import (  # type: ignore
            SimulationRun,
            TerminationReason,
        )
        from tau2.evaluator.evaluator import (  # type: ignore
            EvaluationType,
            evaluate_simulation,
        )
        from tau2.orchestrator.orchestrator import Role  # type: ignore
        from tau2.utils.utils import get_now  # type: ignore

        if self.trajectory:
            last_message = self.trajectory[-1]
            if isinstance(last_message, AssistantMessage):
                to_role = Role.USER if not last_message.is_tool_call() else Role.ENV
            elif isinstance(last_message, UserMessage):
                to_role = Role.AGENT if not last_message.is_tool_call() else Role.ENV
            else:
                to_role = (
                    Role.AGENT if last_message.requestor == "assistant" else Role.USER
                )
        else:
            first_message = _default_first_agent_message()
            self.trajectory.append(first_message)
            to_role = Role.USER
            last_message = first_message

        start_time = datetime.now()
        done = False

        while not done:
            if self.max_steps and self.step_count >= self.max_steps:
                self.termination_reason = TerminationReason.MAX_STEPS
                break
            if self.max_errors and self.num_errors >= self.max_errors:
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
                break

            if to_role == Role.USER:
                user_message = await self._generate_user_message()
                self.trajectory.append(user_message)
                if user_message.is_tool_call():
                    to_role = Role.ENV
                    last_message = user_message
                else:
                    to_role = Role.AGENT
                    last_message = user_message
                if _is_user_stop(user_message.content):
                    self.termination_reason = TerminationReason.USER_STOP
                    done = True
            elif to_role == Role.AGENT:
                agent_message = await self._generate_agent_message()
                self.trajectory.append(agent_message)
                if agent_message.is_tool_call():
                    to_role = Role.ENV
                    last_message = agent_message
                else:
                    to_role = Role.USER
                    last_message = agent_message
            else:
                tool_messages = self._execute_tools(last_message)
                self.trajectory.extend(tool_messages)
                if not tool_messages:
                    to_role = (
                        Role.USER if last_message.requestor == "user" else Role.AGENT
                    )
                else:
                    requestor = tool_messages[-1].requestor
                    to_role = Role.USER if requestor == "user" else Role.AGENT
                last_message = tool_messages[-1] if tool_messages else last_message
            self.step_count += 1

        end_time = datetime.now()
        final_reason = self.termination_reason or "completed"
        simulation = SimulationRun(
            id=f"{self.domain}_{self.tau2_task.id}_{uuid4().hex}",
            task_id=self.tau2_task.id,
            timestamp=get_now(),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=(end_time - start_time).total_seconds(),
            termination_reason=final_reason,  # type: ignore[arg-type]
            agent_cost=None,
            user_cost=None,
            messages=self.trajectory,
            trial=self.trial,
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=self.tau2_task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.domain,
        )
        return TauBenchResult(
            simulation=simulation,
            reward_info=reward_info,
            termination_reason=str(final_reason),
        )

    async def _generate_agent_message(self):
        from tau2.data_model.message import (
            AssistantMessage,  # type: ignore
        )

        messages = [ChatMessageSystem(content=self.agent_system_prompt)]
        messages.extend(_tau2_to_agent_history(self.trajectory))
        response = await self.agent_model.generate(
            messages,
            tools=self.agent_tools,
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
        return AssistantMessage(
            role="assistant",
            content=completion,
            tool_calls=tool_calls,
            raw_data=response.model_dump(mode="json"),
        )

    async def _generate_user_message(self):
        from tau2.data_model.message import UserMessage  # type: ignore

        messages = [ChatMessageSystem(content=self.user_system_prompt)]
        messages.extend(_tau2_to_user_history(self.trajectory))
        response = await self.user_model.generate(
            messages,
            tools=self.user_tools,
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
        return UserMessage(
            role="user",
            content=completion,
            tool_calls=tool_calls,
            raw_data=response.model_dump(mode="json"),
        )

    def _execute_tools(self, message) -> list:
        from tau2.data_model.message import ToolMessage  # type: ignore

        tool_messages: list[ToolMessage] = []
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
    user_model: str = "openai/gpt-4.1-mini",
    max_steps: int = 200,
    max_errors: int = 10,
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

        runner = TauBenchRunner(
            domain=str(domain),
            task_payload=task_payload,
            trial=trial,
            agent_model=candidate,
            user_model=user_model_instance,
            max_steps=max_steps,
            max_errors=max_errors,
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
