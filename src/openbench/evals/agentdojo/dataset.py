from copy import deepcopy
from typing import Literal, TypeVar, Union

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver

from openbench.evals.agentdojo.base_tasks import (
    BaseInjectionTask,
    BaseUserTask,
    TaskEnvironment,
)
from openbench.evals.agentdojo.task_suite import BaseTaskSuite, get_sandbox_path
from openbench.evals.agentdojo.task_suites.banking.environment import BankingEnvironment
from openbench.evals.agentdojo.task_suites.banking.task_suite import BANKING_TASK_SUITE
from openbench.evals.agentdojo.task_suites.slack.environment import SlackEnvironment
from openbench.evals.agentdojo.task_suites.slack.task_suite import SLACK_TASK_SUITE
from openbench.evals.agentdojo.task_suites.travel.environment import TravelEnvironment
from openbench.evals.agentdojo.task_suites.travel.task_suite import TRAVEL_TASK_SUITE
from openbench.evals.agentdojo.task_suites.workspace.environment import (
    WorkspaceEnvironment,
)
from openbench.evals.agentdojo.task_suites.workspace.task_suite import (
    WORKSPACE_TASK_SUITE,
)
from openbench.evals.agentdojo.task_suites.workspace_plus.environment import (
    WorkspacePlusEnvironment,
)
from openbench.evals.agentdojo.task_suites.workspace_plus.task_suite import (
    WORKSPACE_PLUS_TASK_SUITE,
)
from openbench.evals.agentdojo.tools.terminal import run_bash_command

TaskSuite = Union[
    BaseTaskSuite[BankingEnvironment],
    BaseTaskSuite[WorkspacePlusEnvironment],
    BaseTaskSuite[SlackEnvironment],
    BaseTaskSuite[TravelEnvironment],
    BaseTaskSuite[WorkspaceEnvironment],
]

TASK_SUITES: dict[str, TaskSuite] = {
    "banking": BANKING_TASK_SUITE,
    "workspace_plus": WORKSPACE_PLUS_TASK_SUITE,
    "slack": SLACK_TASK_SUITE,
    "travel": TRAVEL_TASK_SUITE,
    "workspace": WORKSPACE_TASK_SUITE,
}


T = TypeVar("T", bound=TaskEnvironment)


def make_user_task_sample(user_task: BaseUserTask[T], task_suite_name: str) -> Sample:
    sandbox = None
    if user_task.REQUIRES_SANDBOX:
        sandbox_path = get_sandbox_path(task_suite_name)
        if sandbox_path is None:
            raise ValueError(
                f"Sandbox path not found for task suite {task_suite_name}. "
                "Please check the task suite configuration."
            )
        sandbox = ("docker", sandbox_path)
    return Sample(
        input=user_task.PROMPT,
        id=f"{task_suite_name}-u{user_task.ID}",
        metadata={
            "task_suite_name": task_suite_name,
            "task_id": user_task.ID,
            "requires_sandbox": user_task.REQUIRES_SANDBOX,
        },
        sandbox=sandbox,
    )


def make_injection_task_sample(
    user_task: BaseUserTask[T],
    injection_task: BaseInjectionTask[T],
    task_suite_name: str,
) -> Sample:
    sandbox = None
    if injection_task.REQUIRES_SANDBOX or user_task.REQUIRES_SANDBOX:
        sandbox_path = get_sandbox_path(task_suite_name)
        if sandbox_path is None:
            raise ValueError(
                f"Sandbox path not found for task suite {task_suite_name}. "
                "Please check the task suite configuration."
            )
        sandbox = ("docker", sandbox_path)
    return Sample(
        input=user_task.PROMPT,
        id=f"{task_suite_name}-u{user_task.ID}-i{injection_task.ID}",
        metadata={
            "task_suite_name": task_suite_name,
            "task_id": user_task.ID,
            "injection_task_id": injection_task.ID,
            "requires_sandbox": user_task.REQUIRES_SANDBOX
            or injection_task.REQUIRES_SANDBOX,
        },
        sandbox=sandbox,
    )


def _process_task_suite(
    suite: BaseTaskSuite[T],
    task_suite_name: str,
    with_sandbox_tasks: Literal["yes", "no", "only"],
    user_task_ids: list[int] | None,
    injection_task_ids: list[int] | None,
    with_injections: bool,
) -> list[Sample]:
    samples = []
    for user_task in suite.user_tasks.values():
        if user_task_ids is not None and int(user_task.ID) not in user_task_ids:
            continue
        if with_sandbox_tasks == "only" and not user_task.REQUIRES_SANDBOX:
            continue
        if with_sandbox_tasks == "no" and user_task.REQUIRES_SANDBOX:
            continue
        if not with_injections:
            samples.append(make_user_task_sample(user_task, task_suite_name))
            continue
        for injection_task in suite.injection_tasks.values():
            if (
                injection_task_ids is not None
                and int(injection_task.ID) not in injection_task_ids
            ):
                continue
            if with_sandbox_tasks == "only" and not injection_task.REQUIRES_SANDBOX:
                continue
            if with_sandbox_tasks == "no" and injection_task.REQUIRES_SANDBOX:
                continue
            samples.append(
                make_injection_task_sample(user_task, injection_task, task_suite_name)
            )
    return samples


def load_dataset(
    with_sandbox_tasks: Literal["yes", "no", "only"] = "yes",
    task_suite: list[str] | str | None = None,
    user_task_ids: list[int] | None = None,
    injection_task_ids: list[int] | None = None,
    with_injections: bool = True,
) -> MemoryDataset:
    if task_suite is None:
        task_suites = {
            name: suite for name, suite in TASK_SUITES.items() if name != "workspace"
        }
    elif isinstance(task_suite, str):
        task_suites = {task_suite: TASK_SUITES[task_suite]}
    elif isinstance(task_suite, list):
        task_suites = {name: TASK_SUITES[name] for name in task_suite}
    if "workspace_plus" in task_suites and "workspace" in task_suites:
        raise ValueError(
            "Both 'workspace_plus' and 'workspace' task suites are present. "
            "Please specify only one of them."
        )

    if (user_task_ids is not None or injection_task_ids is not None) and len(
        task_suites
    ) > 1:
        raise ValueError(
            "Cannot filter by task id when multiple task suites are provided. "
            "Please specify only one task suite."
        )

    samples = []
    for task_suite_name, suite in task_suites.items():
        # Type narrowing for each specific task suite type
        if isinstance(suite, BaseTaskSuite):
            samples.extend(
                _process_task_suite(
                    suite,
                    task_suite_name,
                    with_sandbox_tasks,
                    user_task_ids,
                    injection_task_ids,
                    with_injections,
                )
            )
        else:
            raise TypeError(
                f"Expected BaseTaskSuite instance for {task_suite_name}, "
                f"got {type(suite).__name__}"
            )

    return MemoryDataset(samples=samples, name="AgentDojo")


@solver
def setup_environment(injections: str | dict[str, str]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_suite = TASK_SUITES[state.metadata["task_suite_name"]]
        if state.metadata["requires_sandbox"]:
            state.tools = task_suite.tools + [run_bash_command()]
        else:
            state.tools = task_suite.tools

        default_injections = task_suite.get_injection_vector_defaults()

        if "injection_task_id" in state.metadata:
            state.metadata["injection_task"] = task_suite.injection_tasks[
                int(state.metadata["injection_task_id"])
            ]

            if isinstance(injections, str):
                attack_str = injections.format(
                    goal=state.metadata["injection_task"].GOAL
                )
                for injection_target in default_injections:
                    default_injections[injection_target] = attack_str
            else:
                for injection_target in default_injections:
                    if injection_target in injections:
                        default_injections[injection_target] = injections[
                            injection_target
                        ].format(goal=state.metadata["injection_task"].GOAL)

        environment = task_suite.load_and_inject_default_environment(default_injections)
        state.metadata["task_suite"] = task_suite
        state.metadata["pre_environment"] = deepcopy(environment)
        state.metadata["user_task"] = task_suite.user_tasks[
            int(state.metadata["task_id"])
        ]
        # Loop over items in environment and store them
        for key in type(environment).model_fields.keys():
            state.store.set(key, getattr(environment, key))
        return state

    return solve
