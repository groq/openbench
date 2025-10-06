from typing import cast

from openbench.evals.agentdojo.task_suite import BaseTaskSuite
from openbench.evals.agentdojo.task_suites.workspace.task_suite import (
    INJECTION_TASKS as WORKSPACE_INJECTION_TASKS,
)
from openbench.evals.agentdojo.task_suites.workspace.task_suite import (
    USER_TASKS as WORKSPACE_USER_TASKS,
)
from openbench.evals.agentdojo.task_suites.workspace.task_suite import (
    WORKSPACE_TASK_SUITE,
)
from openbench.evals.agentdojo.task_suites.workspace_plus.environment import (
    WorkspacePlusEnvironment,
)
from openbench.evals.agentdojo.task_suites.workspace_plus.injection_tasks import (
    InjectionTask14,
    WorkspacePlusInjectionTask,
)
from openbench.evals.agentdojo.task_suites.workspace_plus.user_tasks import (
    UserTask40,
    UserTask41,
    WorkspacePlusUserTask,
)

WorkspacePlusTaskSuite = BaseTaskSuite[WorkspacePlusEnvironment]

TOOLS = WORKSPACE_TASK_SUITE.tools


USER_TASKS: list[type[WorkspacePlusUserTask]] = cast(
    list[type[WorkspacePlusUserTask]], WORKSPACE_USER_TASKS
) + [
    UserTask40,
    UserTask41,
]

INJECTION_TASKS: list[type[WorkspacePlusInjectionTask]] = cast(
    list[type[WorkspacePlusInjectionTask]], WORKSPACE_INJECTION_TASKS
) + [InjectionTask14]

WORKSPACE_PLUS_TASK_SUITE = WorkspacePlusTaskSuite(
    name="workspace_plus",
    environment_type=WorkspacePlusEnvironment,
    tools=TOOLS,
    user_tasks=USER_TASKS,
    injection_tasks=INJECTION_TASKS,
)
