from inspect_ai import Task, task
from tau_bench.envs import get_env
from tau_bench.run import agent_factory
from tau_bench.types import EnvRunResult, RunConfig

from typing import Optional, List


class TmpConfig:
    model_provider: str
    user_model_provider: str = "openai"
    model_provider: str = "openai"
    model: str = "gpt-4o"
    user_model: str = "gpt-4o"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    temperature: float = 1.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: Optional[List[int]] = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: Optional[str] = None


tmp_config = TmpConfig()


@task
def taubench():
    env = get_env(
        env_name=tmp_config.env,
        user_strategy="llm",
        user_model=tmp_config.user_model,
        user_provider=tmp_config.user_model_provider,
        task_split=tmp_config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tmp_config,
    )

    res = agent.solve(
        env=env,
        task_index=0,
    )

    res_info = EnvRunResult(
      task_id=0,
      reward=res.reward,
      info=res.info,
      traj=res.messages,
      trial=1,
    )

    print(res_info)

    return Task(
        name="taubench",
        # dataset=get_taubench_dataset(),
        # scorer=get_taubench_scorer(),
    )
