# Ported from https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentdojo
# See LICENSE

from typing import Any, Literal

from inspect_ai import Task, task

from openbench.evals.agentdojo.agents.agent import get_agent
from openbench.evals.agentdojo.attacks.attack import get_attack
from openbench.evals.agentdojo.scorer import injection_task_scorer, user_task_scorer


@task
def agentdojo(
    # attack configuration
    attack: str = "important_instructions",
    attack_kwargs: dict[str, Any] | None = None,
    # agent configuration
    agent: str = "default",
    agent_kwargs: dict[str, Any] | None = None,
    # dataset selection
    with_sandbox_tasks: Literal["yes", "no", "only"] = "yes",
    workspace: list[str] | str | None = None,
    user_task_ids: list[int] | None = None,
    injection_task_ids: list[int] | None = None,
    with_injections: bool = True,
) -> Task:
    """
    AgentDojo benchmark task.

    Args:
        attack (str): The name of the injection attack to use.
        attack_kwargs (dict[str, Any]): Additional arguments to pass to the attack.

        agent (str): The name of the solver to use for the agent. We provide a default agent if unspecified.
        agent_kwargs: Additional arguments to pass to the agent.

        with_sandbox_tasks (Literal['yes', 'no', 'only']): Whether to include sandbox tasks in the dataset.
        workspace (list[str] | str | None): The task suite to use for the dataset. If None, all task suites are used.
        user_task_ids (list[int] | None): List of user task IDs to include in the dataset. If None, all user tasks are included.
        injection_task_ids (list[int] | None): List of injection task IDs to include in the dataset. If None, all injection tasks are included.
        with_injections (bool): Whether to include injections in the dataset.
    """
    # Importing inside the function to avoid importing pydantic with email-validator at import time,
    # used by Inbox via WorkspaceEnvironment and TravelEnvironment
    from openbench.evals.agentdojo.dataset import load_dataset, setup_environment

    dataset = load_dataset(
        with_sandbox_tasks=with_sandbox_tasks,
        task_suite=workspace,
        user_task_ids=user_task_ids,
        injection_task_ids=injection_task_ids,
        with_injections=with_injections,
    )

    injections = get_attack(attack_name=attack, **(attack_kwargs or {}))

    return Task(
        dataset=dataset,
        setup=setup_environment(injections=injections),
        solver=get_agent(agent=agent, **(agent_kwargs or {})),
        scorer=injection_task_scorer() if with_injections else user_task_scorer(),
    )
