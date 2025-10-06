import abc
import enum
from typing import Any, Generic, TypeVar

from inspect_ai.tool import ToolCall
from pydantic import BaseModel
from typing_extensions import Self


class TaskDifficulty(enum.Enum):
    EASY = enum.auto()
    MEDIUM = enum.auto()
    HARD = enum.auto()


class TaskEnvironment(BaseModel):
    """Base class for the environment in which the agent is running."""

    @classmethod
    @abc.abstractmethod
    def create(cls, data: dict[str, Any]) -> Self:
        """Create the environment from a dictionary."""
        ...


Env = TypeVar("Env", bound="TaskEnvironment")


class BaseUserTask(abc.ABC, Generic[Env]):
    ID: str
    """The ID of the task. This should be unique across all tasks. This is set automatically by the task suite with `register_user_task`."""
    PROMPT: str
    """The prompt that the model receives. It must be precise and LLM-understandable."""
    GROUND_TRUTH_OUTPUT: str = ""
    """The portion of output that the model should return when the ground truth is executed."""
    DIFFICULTY: TaskDifficulty = TaskDifficulty.EASY
    """The difficulty of the task. This is used to determine the difficulty of the task in the task suite."""
    REQUIRES_SANDBOX: bool = False
    """Whether the task requires spinning up a sandbox."""

    @staticmethod
    def init_environment(environment: Env) -> Env:
        """Use this function if the environment needs to be initialized. By default, nothing is done."""
        return environment

    @abc.abstractmethod
    def ground_truth(self, pre_environment: Env) -> list[ToolCall]:
        """Returns a list of [`ToolCall`][agentdojo.functions_runtime.ToolCall]s that solve the task if executed.

        Must be implemented by the subclass.

        Each [`ToolCall`][agentdojo.functions_runtime.ToolCall] is a dictionary with the following keys:
            - `function`: the function name
            - `args`: a dict of arguments, with the argument name as the key and the argument value as the value.
              If the value is unknown and not important for solving the task, you can use any value.

        Args:
            pre_environment: the environment before the function calls are executed. Can be used to retrieve some
                arguments that can be used for the function calls.
        """
        ...

    @abc.abstractmethod
    async def utility(
        self,
        model_output: str,
        pre_environment: Env,
        post_environment: Env,
        traces: list[ToolCall],
        strict: bool = True,
    ) -> bool:
        """Checks if the task was executed correctly. Returns True if the task was executed correctly, False otherwise.

        It can do so by checking the model output, and by looking for differences between the environment before and
        after the agent is executed. It may also use the traces of the functions that are called.

        Must be implemented by the subclass.

        Args:
            model_output: the output of the agent
            pre_environment: the environment before the agent is executed
            post_environment: the environment after the agent is executed
            traces: the traces of the functions that are called
            strict: if `True` the state check will be strict. If `False` the state check will be relaxed.
                This is should be used in read-only tasks when combined with tasks with side-effects.
        """
        ...


class BaseInjectionTask(abc.ABC, Generic[Env]):
    ID: str
    """The ID of the task. This should be unique across all tasks. This is set automatically by the task suite with `register_injection_task`."""
    GOAL: str
    """The goal of the injection. It must be precise and LLM-understandable. It should be usable to generate the injection prompts by attacks."""
    DIFFICULTY: TaskDifficulty = TaskDifficulty.EASY
    """The difficulty of the task. This is used to determine the difficulty of the task in the task suite."""
    GROUND_TRUTH_OUTPUT: str = ""
    """The portion of output that the model should return when the ground truth is executed."""
    REQUIRES_SANDBOX: bool = False
    """Whether the task requires spinning up a sandbox."""

    @abc.abstractmethod
    def ground_truth(self, pre_environment: Env) -> list[ToolCall]:
        """Return a list of function calls that correctly execute the injection task if executed.

        Must be implemented by the subclass.

        Each function call is a dictionary with the following keys:
            - `function`: the function name
            - `args`: a dict of arguments, with the argument name as the key and the argument value as the value.
              If the value is unknown and not important for solving the task, use any value.

        Args:
            pre_environment: the environment before the function calls are executed. Can be used to retrieve some
                arguments that can be used for the function calls.
        """
        ...

    @abc.abstractmethod
    def placeholder_ground_truth(self, pre_environment: Env) -> list[ToolCall]:
        """Return a list of function calls that correctly execute the injection task if executed.

        Must be implemented by the subclass.

        Each function call is a dictionary with the following keys:
            - `function`: the function name
            - `args`: a dict of arguments, with the argument name as the key and the argument value as the value.
              If the value is unknown and not important for solving the task, use any value.

        Args:
            pre_environment: the environment before the function calls are executed. Can be used to retrieve some
                arguments that can be used for the function calls.
        """
        ...

    @abc.abstractmethod
    async def security(
        self,
        model_output: str,
        pre_environment: Env,
        post_environment: Env,
        traces: list[ToolCall],
    ) -> bool:
        """Checks if the injection goal was executed correctly. Returns True if the injection was executed correctly, False otherwise.

        It can do so by checking the model output and by looking for differences between the environment before and
        after the agent is executed. It may also use the traces of the functions that are called.

        Must be implemented by the subclass.

        Args:
            model_output: the output of the agent
            pre_environment: the environment before the agent is executed
            post_environment: the environment after the agent is executed
            traces: the traces of the functions that are called
        """
        ...
