"""
Machine Translation from One Book (MTOB)

Garrett Tanzer, Mirac Suzgun, Eline Visser,
Dan Jurafsky, Luke Melas-Kyriazi

https://arxiv.org/abs/2309.16575

Based on https://github.com/meta-llama/llama-stack-evals

# run eval
export MTOB_KEY="mtob-eval-encode" # or use SET if using cmd for Windows
bench eval mtob --model "groq/llama-3.1-8b-versatile"
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import TaskState, solver, Generate
from inspect_ai.model import ChatMessageUser
from dotenv import load_dotenv

from openbench.scorers.mtob import custom_scorer
from openbench.datasets.mtob import (
    get_llamastack_mtob_dataset,
    get_groq_mtob_dataset,
    create_mtob_user_prompt_for_groq_ds,
)


load_dotenv()


DEBUG = False
CURR_TASK = {
    "translation_task": None,
    "provider": None,
    "knowledge_base_task": None,
}


@solver
def custom_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata["provider"] == "groq":
            if state.metadata["knowledge_base_task"] == "claude-book-medium":
                USER_PROMPT = create_mtob_user_prompt_for_groq_ds(
                    source=state.input,
                    translation_direction=state.metadata["translation_direction"],
                    context_type=state.metadata["knowledge_base_task"],
                )

            elif state.metadata["knowledge_base_task"] == "claude-book-long":
                USER_PROMPT = create_mtob_user_prompt_for_groq_ds(
                    source=state.input,
                    translation_direction=state.metadata["translation_direction"],
                    context_type=state.metadata["knowledge_base_task"],
                )

            elif state.metadata["knowledge_base_task"] == "zero-shot":
                USER_PROMPT = create_mtob_user_prompt_for_groq_ds(
                    source=state.input,
                    translation_direction=state.metadata["translation_direction"],
                    context_type=state.metadata["knowledge_base_task"],
                )

            else:
                raise ValueError(
                    f"Invalid knowledge base task: {state.metadata['knowledge_base_task']}"
                )

            resp = await model.generate(
                input=[
                    ChatMessageUser(content=USER_PROMPT),
                ],
            )
            resp_text = resp.choices[0].message.content
            if "Kalamang translation:" in resp_text:
                resp_text = resp_text.split("Kalamang translation:")[1].strip()
            if "English translation:" in resp_text:
                resp_text = resp_text.split("English translation:")[1].strip()
            state.messages.append(ChatMessageUser(content=resp_text))

        elif state.metadata["provider"] == "llamastack":
            resp = await model.generate(
                input=[
                    ChatMessageUser(content=state.input),  # type: ignore
                ],
            )
            state.messages.append(resp.choices[0].message)

        return state

    return solve


@task
def mtob(subtask: str, cot: bool = False) -> Task:
    """
    subtasks: comma separated list of subtasks
      - ek/groq/claude-book-medium
      - ek/groq/claude-book-long
      - ek/llamastack/half-book
      - ek/llamastack/full-book
      - ek/groq/zero-shot
      - ke/groq/claude-book-medium
      - ke/groq/claude-book-long
      - ke/groq/zero-shot
      - ke/llamastack/half-book
      - ke/llamastack/full-book
    """

    translation_task = subtask.split("/")[0]
    provider = subtask.split("/")[1]
    knowledge_base_task = subtask.split("/")[2]

    # global CURR_TASK
    CURR_TASK = {
        "translation_task": translation_task,
        "provider": provider,
        "knowledge_base_task": knowledge_base_task,
    }

    if provider == "llamastack":
        dataset = get_llamastack_mtob_dataset(subtask=subtask, curr_task=CURR_TASK)
    elif provider == "groq":
        dataset = get_groq_mtob_dataset(curr_task=CURR_TASK)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    return Task(
        name="mtob",
        dataset=dataset,
        solver=custom_solver(),
        scorer=custom_scorer(),
        config=GenerateConfig(),
    )
