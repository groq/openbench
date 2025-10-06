from typing import Generic, TypeVar

from inspect_ai.tool import ToolCall

from openbench.evals.agentdojo.base_tasks import (
    BaseUserTask,
    TaskDifficulty,
    TaskEnvironment,
)

ENV = TypeVar("ENV", bound=TaskEnvironment)


class TaskCombinator(Generic[ENV]):
    def create_combined_task(
        self,
        new_task_name: str,
        user_task_1: BaseUserTask[ENV],
        user_task_2: BaseUserTask[ENV],
        prompt: str | None = None,
    ) -> type[BaseUserTask[ENV]]:
        task_type = type(user_task_1)  # Capture the type of the first task

        attributes = {
            "DIFFICULTY": TaskDifficulty.MEDIUM,
            "GROUND_TRUTH_OUTPUT": f"1. {user_task_1.GROUND_TRUTH_OUTPUT} 2. {user_task_2.GROUND_TRUTH_OUTPUT}",
            "PROMPT": prompt
            if prompt
            else f"Perform the following tasks in the order they are given: 1. {user_task_1.PROMPT} 2. {user_task_2.PROMPT}",
        }

        def ground_truth(
            self: BaseUserTask[ENV], pre_environment: ENV
        ) -> list[ToolCall]:
            return user_task_1.ground_truth(pre_environment) + user_task_2.ground_truth(
                pre_environment
            )

        async def utility(
            self: BaseUserTask[ENV],
            model_output: str,
            pre_environment: ENV,
            post_environment: ENV,
            traces: list[ToolCall],
            strict: bool = True,
        ) -> bool:
            return (
                await user_task_1.utility(
                    model_output,
                    pre_environment,
                    post_environment,
                    traces,
                    strict=False,
                )
            ) and (
                await user_task_2.utility(
                    model_output, pre_environment, post_environment, traces, strict=True
                )
            )

        attributes["ground_truth"] = ground_truth
        attributes["utility"] = utility

        # Dynamically create the new combined task class
        NewCombinedTask = type(new_task_name, (task_type,), attributes)

        return NewCombinedTask
