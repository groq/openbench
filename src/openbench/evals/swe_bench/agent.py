"""
SWE-Bench agent implementation using SWE-Agent tools.
"""

import logging
from typing import List, Dict, Optional

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import solver, TaskState, Generate, Solver
from inspect_ai.tool import Tool, bash, tool
from inspect_ai.util import store

from .tools import (
    sweagent_tooldefs,
    setup_sweagent_environment,
)

logger = logging.getLogger(__name__)

BASH_TOOL_TIMEOUT = 60 * 5

# Default SWE-Agent bundles from config/default.yaml
DEFAULT_BUNDLES: List[str] = [
    "defaults",  # Core file navigation
    "search",  # File search tools
    "edit_replace",  # Search and replace editing
]


@solver
def swe_bench_agent(
    token_limit: int = 1_000_000,
    bundles: Optional[List[str]] = None,
    window: int = 100,
    overlap: int = 2,
) -> Solver:
    """
    A solver that uses SWE-Agent tools to solve SWE-Bench tasks.

    Args:
        token_limit: Maximum number of tokens for the conversation.
        bundles: List of SWE-Agent bundles to include.
        window: Size of the file viewing window (lines).
        overlap: Number of lines to overlap when scrolling.

    Returns:
        A solver configured with SWE-Agent tools.
    """
    include_bundles = bundles or DEFAULT_BUNDLES

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize submission tracking
        store().set("submitted_answer", None)

        # Set token limit
        state.token_limit = token_limit

        # Setup environment and load tools
        await setup_sweagent_environment(window=window, overlap=overlap)
        swe_agent_tools = await sweagent_tooldefs(include_bundles=include_bundles)

        # Flatten tool list
        all_tools = []
        for bundle_tools in swe_agent_tools.values():
            all_tools.extend([tool_def.as_tool() for tool_def in bundle_tools])

        # Configure tools
        state.tools = [bash(timeout=BASH_TOOL_TIMEOUT), submit_answer()] + all_tools
        state.tool_choice = "auto"

        # Create initial prompt
        state.messages = [
            ChatMessageUser(
                content=create_initial_prompt(
                    question=state.user_prompt.text,
                    token_limit=token_limit,
                    repo=state.metadata["repo"],
                    swe_agent_tools=swe_agent_tools,
                    window_size=window,
                    overlap_size=overlap,
                ),
            ),
        ]

        # Main interaction loop
        while state.token_usage < token_limit:
            state = await generate(state, tool_calls="loop")

            # Check if answer was submitted
            if store().get("submitted_answer") is not None:
                break

            # Continue working
            message = (
                f"You have used {state.token_usage:,} tokens (hard limit: {token_limit:,} tokens). "
                f"Continue working on the problem. Once you've made the necessary changes to the "
                f"repository, submit your answer using the `submit_answer` tool."
            )
            state.messages.append(ChatMessageUser(content=message))

        return state

    return solve


@tool
def submit_answer() -> Tool:
    """Tool to signal completion of changes to the repository."""

    async def execute() -> str:
        """
        Call this tool to signal that you have finished making changes to the repository to solve
        the task. The tool does not take any arguments.
        """
        store().set("submitted_answer", True)
        return "Your answer has been recorded. No feedback is provided."

    return execute


def create_initial_prompt(
    question: str,
    token_limit: int,
    repo: str,
    swe_agent_tools: Dict[str, List],
    window_size: int,
    overlap_size: int,
) -> str:
    """Generate the initial prompt with tool descriptions."""

    # Format tool descriptions
    tool_descriptions = ""
    for bundle, tools in swe_agent_tools.items():
        tool_names = ", ".join(
            f"`{tool.name}`" for tool in sorted(tools, key=lambda t: t.name)
        )
        tool_descriptions += f"\n      - {bundle}: {tool_names}"

    return f"""
You will be solving an open issue from the {repo} repository. You have access to a Python environment
containing the repository, and you will need to modify the code to solve the issue. 
When you are done, call the `submit_answer` tool to let us know.

* Environment:
    - You have access to a Python environment containing the repository. You interact with the 
      environment by calling various tools.
    - The environment does not have internet access.

* Token limits:
    - There is a hard limit of {token_limit:,} tokens. If you exceed this, the conversation will end
      immediately (even if you haven't called the submit tool). The final state of the repository at
      that point will be used as your answer. 
    - Both input and output tokens count towards the limit.

* SWE-Agent Tools:
   - Available tools by category: {tool_descriptions}
   - Additional guidance on the file viewing and editing tools:
      - The file viewing and editing tools let you view {window_size} lines at a time with an overlap of {overlap_size} lines.
      - You can use the `scroll_up` and `scroll_down` commands to navigate through larger files.
      - Avoid using the scrolling commands multiple times. Instead, use the `goto` command to jump to a specific line, or
        open the file at a specific line.

* You also have access to the `bash` tool:
   - You can use the tool to execute arbitrary shell commands in the repository environment.
   - The tool has a timeout of {BASH_TOOL_TIMEOUT} seconds.

* The `submit_answer` tool takes no arguments and simply records that you have finished making changes to the repository.

Here is the issue you need to solve:
<issue>
{question}
</issue>
"""
