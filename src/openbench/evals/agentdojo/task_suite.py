import importlib.resources
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from inspect_ai.tool import Tool
from pydantic import BaseModel

from openbench.evals.agentdojo.base_tasks import (
    BaseInjectionTask,
    BaseUserTask,
    TaskEnvironment,
)
from openbench.evals.agentdojo.yaml_loader import ImportLoader


class InjectionVector(BaseModel):
    description: str
    default: str


Env = TypeVar("Env", bound=TaskEnvironment, covariant=True)


@lru_cache
def read_suite_file(suite_name: str, file: str, suite_data_path: Path | None) -> str:
    if suite_data_path is not None:
        path = suite_data_path / file
        with path.open() as f:
            data_yaml = yaml.load(f, ImportLoader)
        return yaml.dump(data_yaml, default_flow_style=False)
    package_files = importlib.resources.files("openbench.evals")
    with importlib.resources.as_file(package_files) as p:
        file_path = os.path.join(p, "agentdojo", "data", "suites", suite_name, file)
    with open(file_path) as f:
        # Load into yaml to resolve imports
        data_yaml = yaml.load(f, ImportLoader)
    # dump back to string to pass to include injections later
    return yaml.dump(data_yaml, default_flow_style=False)


@lru_cache
def get_sandbox_path(
    suite_name: str, suite_data_path: Path | None = None
) -> str | None:
    if suite_data_path is not None:
        # check if terminal is in the path
        path = Path(suite_data_path / "terminal")
    else:
        package_files = importlib.resources.files("openbench.evals")
        with importlib.resources.as_file(package_files) as p:
            path = p / "agentdojo" / "data" / "suites" / suite_name / "terminal"

    if path.exists():
        return str(path / "compose.yaml")

    return None


def validate_injections(
    injections: dict[str, str], injection_vector_defaults: dict[str, str]
) -> None:
    injections_set = set(injections.keys())
    injection_vector_defaults_set = set(injection_vector_defaults.keys())
    if not injections_set.issubset(injection_vector_defaults_set):
        raise ValueError("Injections must be a subset of the injection vector defaults")


class BaseTaskSuite(Generic[Env]):
    """
    A suite of tasks that can be run in an environment.

    Tasks can be both user tasks and injection tasks. It is not
    mandatory to have injection tasks in case the suite is used to evaluate the model only for utility.

    Args:
        name: The name of the suite.
        environment_type: The environment type that the suite operates on.
        tools: A list of tools that the agent can use to solve the tasks.
        data_path: The path to the suite data directory. It should be provided for non-default suites.
            The directory should contain the following files:
                - `environment.yaml`: The data of the environment.
                - `injection_vectors.yaml`: The injection vectors in the environment.
        user_tasks: A list of user tasks in the suite.
        injection_tasks: A list of injection tasks in the task.
    """

    def __init__(
        self,
        name: str,
        environment_type: type[Env],
        tools: list[Tool],
        data_path: str = "",
        user_tasks: list[type[BaseUserTask[Env]]] = [],
        injection_tasks: list[type[BaseInjectionTask[Env]]] = [],
    ):
        self.name = name
        self.environment_type = environment_type
        self.tools = tools
        self.data_path = data_path
        self._register_user_tasks(user_tasks)
        self._register_injection_tasks(injection_tasks)

    def _get_task_number(
        self,
        task_cls: type[BaseUserTask[Env]] | type[BaseInjectionTask[Env]],
        prefix: str,
    ) -> int:
        match = re.match(rf"{prefix}(\d+)", task_cls.__name__)
        if not match:
            raise ValueError(
                f"User tasks must be named {prefix} followed by a number, got {task_cls.__name__}"
            )
        return int(match.group(1))

    def _register_user_tasks(self, user_tasks: list[type[BaseUserTask[Env]]]) -> None:
        """Parses the user task IDs and registers them in the suite."""
        self.user_tasks: dict[int, BaseUserTask[Env]] = {}
        for user_task in user_tasks:
            task_number = self._get_task_number(user_task, "UserTask")
            user_task.ID = str(task_number)
            if task_number in self.user_tasks:
                raise ValueError(f"Duplicate user task ID {task_number}")
            self.user_tasks[task_number] = user_task()

    def _register_injection_tasks(
        self, injection_tasks: list[type[BaseInjectionTask[Env]]]
    ) -> None:
        """Parses the injection task IDs and registers them in the suite."""
        self.injection_tasks: dict[int, BaseInjectionTask[Env]] = {}
        for injection_task in injection_tasks:
            task_number = self._get_task_number(injection_task, "InjectionTask")
            injection_task.ID = str(task_number)
            if task_number in self.injection_tasks:
                raise ValueError(f"Duplicate injection task ID {task_number}")
            self.injection_tasks[task_number] = injection_task()

    def load_and_inject_default_environment(self, injections: dict[str, str]) -> Env:
        environment_text = read_suite_file(
            self.name,
            "environment.yaml",
            Path(self.data_path) if self.data_path else None,
        )
        injection_vector_defaults = self.get_injection_vector_defaults()
        injections_with_defaults = dict(injection_vector_defaults, **injections)
        validate_injections(injections, injection_vector_defaults)
        injected_environment = environment_text.format(**injections_with_defaults)
        environment = self.environment_type.create(yaml.safe_load(injected_environment))
        return environment

    def get_injection_vector_defaults(self) -> dict[str, str]:
        injection_vectors_text = read_suite_file(
            self.name,
            "injection_vectors.yaml",
            Path(self.data_path) if self.data_path else None,
        )
        injection_vectors = yaml.safe_load(injection_vectors_text)
        vectors = {
            vector_id: InjectionVector.model_validate(vector_info)
            for vector_id, vector_info in injection_vectors.items()
        }
        return {vector_id: vector.default for vector_id, vector in vectors.items()}
