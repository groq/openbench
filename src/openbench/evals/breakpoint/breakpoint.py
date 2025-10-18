"""
Breakpoint - Code repair benchmark for evaluating LLM reasoning in debugging

Breakpoint tests LLMs' ability to diagnose, explore, and repair code in Python/Pytest
repositories. It provides two evaluation modes:

- **Remove mode**: Function bodies are deleted; the model must reconstruct them
- **Discovery mode**: Subtle corruptions are introduced; the model must locate and fix them

The benchmark focuses on system-level reasoning by requiring models to:
1. Understand code context within a repository
2. Interpret test failure messages
3. Locate and fix bugs iteratively

Dataset: https://huggingface.co/datasets/uzpg/breakpoint
Paper: http://arxiv.org/pdf/2506.00172

Sample usage:
```bash
# Run remove mode (498 problems) with default budgets (16 tool-use / 4 attempts)
bench eval breakpoint_remove --model "groq/llama-3.1-70b" --limit 10

# Run discovery mode (269 problems)
bench eval breakpoint_discovery --model "groq/llama-3.1-70b" --limit 10

# Run with Docker isolation (slower, but more secure)
bench eval breakpoint_remove --model "groq/llama-3.1-70b" --limit 10 -T use_docker=true

# Run with custom budgets (e.g., for scaling experiments as in paper)
bench eval breakpoint_remove --model "groq/llama-3.1-70b" --limit 10 -T max_tool_uses=32 -T max_attempts=8
```

Citation:
@article{hariharan2025breakpoint,
    title={Breakpoint: Scalable evaluation of system-level reasoning in LLM code agents},
    author={Hariharan, Kaivalya and Girit, Uzay and Wang, Atticus and Andreas, Jacob},
    journal={arXiv preprint arXiv:2506.00172},
    year={2025}
}
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.agent import react
from inspect_ai.solver import solver, Solver, TaskState
from inspect_ai.tool import bash, python
from typing import Any
from pathlib import Path

from openbench.tools.breakpoint_tools import submit_solution
from openbench.scorers.breakpoint_scorer import breakpoint_scorer


# HuggingFace dataset URLs
BREAKPOINT_REMOVE_URL = (
    "https://huggingface.co/datasets/uzpg/breakpoint/resolve/main/data/remove-data.json"
)
BREAKPOINT_DISCOVERY_URL = "https://huggingface.co/datasets/uzpg/breakpoint/resolve/main/data/discovery-data.json"

# Docker compose path for sandbox with network access
TASK_DIR = Path(__file__).parent
COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()


def _get_repo_stat(repo: dict[str, Any], field: str) -> Any:
    """Safely get repo stat from either direct field or stats sub-object.

    Remove mode has direct fields (total_lines, functions_count, etc.)
    Discovery mode has them in repo.stats sub-object.
    """
    # Try direct field first
    if field in repo:
        return repo[field]
    # Try stats sub-object
    if "stats" in repo and isinstance(repo["stats"], dict) and field in repo["stats"]:
        return repo["stats"][field]
    return None


def _extract_metadata(record: dict[str, Any], mode: str) -> dict[str, Any]:
    """Extract all metadata fields from a Breakpoint record.

    This function handles both remove and discovery modes, extracting all
    35 metadata fields including complexity metrics, centrality measures,
    repository statistics, and test information.

    Args:
        record: The dataset record containing repo, test_info, complexity_info, etc.
        mode: Either "remove" or "discovery"

    Returns:
        Dictionary with all metadata fields for the Sample
    """
    repo = record["repo"]
    test_info = record.get("test_info", {})
    complexity_info = record.get("complexity_info", {})
    centrality = complexity_info.get("centrality", {})
    corruption = record.get("corruption", {})

    return {
        # ===== Core identifiers =====
        "mode": mode,
        "repo_name": repo.get("name"),
        "repo_url": repo.get("url"),
        "repo_commit": repo.get("commit"),
        "repo_path": repo.get("path"),
        "repo_code_path": repo.get("code_path"),
        "repo_test_command": repo.get("test_command", "pytest"),
        "repo_exists": repo.get("exists"),
        "repo_final_path": repo.get("repos_final_path"),
        "fpath": record["fpath"],
        "function_name": record["function_name"],
        # ===== Repository statistics =====
        # Handle both direct fields (remove mode) and stats sub-object (discovery mode)
        "repo_files_count": _get_repo_stat(repo, "files_count"),
        "repo_functions_count": _get_repo_stat(repo, "functions_count"),
        "repo_total_lines": _get_repo_stat(repo, "total_lines"),
        "repo_avg_lines_per_file": _get_repo_stat(repo, "avg_lines_per_file"),
        "repo_avg_lines_per_func": _get_repo_stat(repo, "avg_lines_per_func"),
        # ===== Test information =====
        "test_success": test_info.get("success"),
        "test_failed": test_info.get("failed"),
        "test_passed": test_info.get("passed"),
        "test_deselected": test_info.get("deselected"),
        "test_had_execution_error": test_info.get("had_execution_error"),
        "test_error_message": test_info.get("error_message"),
        "baseline_failures": max(test_info.get("failed", 1), 1),  # For scorer
        # ===== Complexity metrics =====
        "cyclomatic_complexity": complexity_info.get("cyclomatic"),
        "line_count": complexity_info.get("line_count"),
        "code_line_count": complexity_info.get("code_line_count"),
        "halstead_volume": complexity_info.get("halstead_volume"),
        "halstead_difficulty": complexity_info.get("halstead_difficulty"),
        # ===== Centrality metrics (graph-based) =====
        "centrality_pagerank": centrality.get("pagerank"),
        "centrality_betweenness": centrality.get("betweenness"),
        "centrality_degree": centrality.get("degree"),
        "centrality_harmonic": centrality.get("harmonic"),
        "centrality_bidirectional_harmonic": centrality.get("bidirectional_harmonic"),
        "centrality_in_degree": centrality.get("in_degree"),
        "centrality_true_in_degree": centrality.get("true_in_degree"),
        "centrality_out_degree": centrality.get("out_degree"),
        "centrality_distance_discount": centrality.get("distance_discount"),
        # ===== Corruption fields =====
        "corruption_score": corruption.get("score") if mode == "discovery" else None,
        "corruption_code": corruption.get("code") if mode == "discovery" else None,
    }


def record_to_sample_remove(record: dict[str, Any]) -> Sample:
    """Convert a remove-mode record to an Inspect AI Sample.

    In remove mode, the function body has been deleted and the model must
    reconstruct it based on:
    - The repository context
    - Test failure messages
    - Function signature and location

    Args:
        record: Dictionary containing repo info, file path, function name,
                test info, and complexity metrics

    Returns:
        Sample with the problem statement and metadata
    """
    repo = record["repo"]
    fpath = record["fpath"]
    function_name = record["function_name"]

    # Simple prompt - agent will clone and explore
    prompt = f"""**Repository**: {repo.get("name", "unknown")}
**Repository URL**: {repo.get("url", "")}
**Commit**: {repo.get("commit", "main")}
**Target File**: {fpath}
**Function**: `{function_name}`

The function `{function_name}` in file `{fpath}` has had its body removed (replaced with `pass`).

**Your task**: Reconstruct the complete, working implementation.

**CRITICAL: You MUST follow these steps IN ORDER**:

**Step 1 - Clone the repository** (REQUIRED FIRST STEP):
   - Use a unique directory name with timestamp to avoid conflicts: `repo_$(date +%s)`
   - Clone command example: `git clone <url> repo_$(date +%s) && cd repo_* && git checkout <commit>`
   - Verify clone succeeded: check for `.git` directory or run `git status`

   DO NOT clone to paths like `/tmp/repo_name` that may already exist.
   DO NOT skip error checking - verify git clone succeeded before continuing.

**Step 2 - Explore and understand** (FOCUS HERE):
   - Read the target file to understand the function signature
   - Read tests to understand requirements
   - Explore related code for context
   - Analyze what the function should do

   DO NOT run the full test suite yet - it will likely fail due to missing dependencies.
   Focus on understanding requirements FIRST, then implement.

**Step 3 - Implement and submit**:
   - Write your complete function implementation
   - Call `submit_solution(code)` with your implementation
   - The tool will handle running tests for you

**Step 4 - If submission fails** (ONLY if needed):
   - Read the test output from submit_solution
   - Install any missing dependencies if tests fail due to imports
   - Refine your solution and submit again (you have 4 attempts total)

**Available tools**:
- `bash()`: Run shell commands (git clone, ls, cat, grep, etc.)
  - **IMPORTANT**: Each bash() call is independent. Always use full paths or cd in the same command.
  - Example: `bash("cd my_repo && ls")` NOT `bash("cd my_repo")` then `bash("ls")`
- `python()`: Execute Python scripts for analysis
- `submit_solution(code)`: Submit your function for testing (max 4 attempts)"""

    return Sample(
        input=prompt,
        target="",  # No ground truth available
        id=f"{repo.get('name', 'unknown')}_{function_name}",
        metadata=_extract_metadata(record, "remove"),
    )


def record_to_sample_discovery(record: dict[str, Any]) -> Sample:
    """Convert a discovery-mode record to an Inspect AI Sample.

    In discovery mode, a subtle corruption has been introduced and the model
    must locate and fix it.

    Args:
        record: Dictionary containing repo info, file path, function name,
                test info, and corruption details

    Returns:
        Sample with the problem statement and metadata
    """
    repo = record["repo"]
    fpath = record["fpath"]
    function_name = record["function_name"]
    corruption = record.get("corruption", {})

    # Simple prompt - agent will clone and explore
    prompt = f"""**Repository**: {repo.get("name", "unknown")}
**Repository URL**: {repo.get("url", "")}
**Commit**: {repo.get("commit", "main")}
**Target File**: {fpath}
**Function**: `{function_name}`

A subtle bug has been introduced in the function `{function_name}`. Here's the corrupted code:

```python
{corruption.get("code", "")}
```

**Your task**: Identify and fix the bug with limited tool calls and submission attempts.

**CRITICAL: You MUST follow these steps IN ORDER**:

**Step 1 - Clone the repository** (REQUIRED FIRST STEP):
   - Use a unique directory name with timestamp to avoid conflicts: `repo_$(date +%s)`
   - Clone command example: `git clone <url> repo_$(date +%s) && cd repo_* && git checkout <commit>`
   - Verify clone succeeded: check for `.git` directory or run `git status`

   DO NOT clone to paths like `/tmp/repo_name` that may already exist.
   DO NOT skip error checking - verify git clone succeeded before continuing.

**Step 2 - Analyze the bug** (FOCUS HERE):
   - The corrupted code is shown above - analyze it carefully
   - Read the target file to see the current implementation
   - Identify what's wrong by comparing with the corrupted code
   - Read related code if needed for context

   DO NOT run the full test suite yet - it will likely fail due to missing dependencies.
   Focus on understanding and fixing the bug FIRST.

**Step 3 - Fix and submit**:
   - Write your corrected implementation
   - Call `submit_solution(code)` with your fixed function
   - The tool will handle running tests for you

**Step 4 - If submission fails** (ONLY if needed):
   - Read the test output from submit_solution
   - Install any missing dependencies if tests fail due to imports
   - Refine your solution and submit again

**Available tools**:
- `bash()`: Run shell commands (git clone, ls, cat, grep, etc.)
  - **IMPORTANT**: Each bash() call is independent. Always use full paths or cd in the same command.
  - Example: `bash("cd my_repo && ls")` NOT `bash("cd my_repo")` then `bash("ls")`
- `python()`: Execute Python scripts for analysis
- `submit_solution(code)`: Submit your fixed function for testing"""

    return Sample(
        input=prompt,
        target="",  # No ground truth available
        id=f"{repo.get('name', 'unknown')}_{function_name}",
        metadata=_extract_metadata(record, "discovery"),
    )


@solver
def setup_metadata_solver(max_attempts: int) -> Solver:
    """Setup max_attempts and sample metadata in store before agent runs.

    This solver initializes the store with max_attempts and copies all
    sample metadata fields to the store so tools can access them.

    Args:
        max_attempts: Maximum submission attempts allowed

    Returns:
        Solver that performs the setup
    """

    async def solve(state: TaskState, generate) -> TaskState:
        from inspect_ai.util import store

        # Initialize store with metadata
        st = store()
        st.set("max_attempts", max_attempts)

        # Copy sample metadata to store for tool access
        if state.metadata:
            for key, value in state.metadata.items():
                st.set(key, value)

        return state

    return solve


@task
def breakpoint_remove(
    max_tool_uses: int = 16,
    max_attempts: int = 4,
    use_docker: bool = False,
) -> Task:
    """
    Breakpoint Remove Mode - Function reconstruction task

    Tests the model's ability to reconstruct deleted function bodies based on:
    - Repository context
    - Test failure messages
    - Function signatures and locations

    Args:
        max_tool_uses: Maximum tool-use iterations (default: 16, as per paper)
                       Set via -T max_tool_uses=N
        max_attempts: Maximum submission attempts (default: 4, as per paper)
                      Set via -T max_attempts=N
        use_docker: Use Docker sandbox with network access (default: False)
                    Set via -T use_docker=true

    Dataset: 498 problems from real Python repositories
    """
    sandbox_config = ("docker", str(COMPOSE_PATH)) if use_docker else "local"

    return Task(
        dataset=json_dataset(
            json_file=BREAKPOINT_REMOVE_URL,
            sample_fields=record_to_sample_remove,
            auto_id=True,
        ),
        setup=setup_metadata_solver(max_attempts),
        solver=react(
            prompt="""You are an expert Python debugging agent.

Follow the instructions in the task description carefully. Use bash and python tools to explore the codebase, understand the requirements, and implement the solution.

When you're confident in your solution, call submit_solution(code) with your complete function implementation.""",
            tools=[
                bash(timeout=180),
                python(timeout=180),
                submit_solution(),
            ],
        ),
        scorer=breakpoint_scorer(),
        sandbox=sandbox_config,
        max_messages=max_tool_uses * 2
        + 1,  # Initial prompt + (each tool call = 2 messages)
    )


@task
def breakpoint_discovery(
    max_tool_uses: int = 16,
    max_attempts: int = 4,
    use_docker: bool = False,
) -> Task:
    """
    Breakpoint Discovery Mode - Bug location and repair task

    Tests the model's ability to locate and fix subtle bugs in code based on:
    - Corrupted function code
    - Test failure messages
    - Repository context

    Args:
        max_tool_uses: Maximum tool-use iterations (default: 16, as per paper)
                       Set via -T max_tool_uses=N
        max_attempts: Maximum submission attempts (default: 4, as per paper)
                      Set via -T max_attempts=N
        use_docker: Use Docker sandbox with network access (default: False)
                    Set via -T use_docker=true

    Dataset: 269 problems from real Python repositories

    In discovery mode, modifications persist throughout the trajectory,
    allowing the agent to iterate and refine solutions.
    """
    sandbox_config = ("docker", str(COMPOSE_PATH)) if use_docker else "local"

    return Task(
        dataset=json_dataset(
            json_file=BREAKPOINT_DISCOVERY_URL,
            sample_fields=record_to_sample_discovery,
            auto_id=True,
        ),
        setup=setup_metadata_solver(max_attempts),
        solver=react(
            prompt="""You are an expert Python debugging agent.

Follow the instructions in the task description carefully. Use bash and python tools to explore the codebase, run tests, and understand the bug.

In discovery mode, your changes persist - you can iterate and refine your solution across multiple attempts.

When you're confident in your fix, call submit_solution(code) with your corrected function implementation.""",
            tools=[
                bash(timeout=180),
                python(timeout=180),
                submit_solution(),
            ],
        ),
        scorer=breakpoint_scorer(),
        sandbox=sandbox_config,
        max_messages=max_tool_uses * 2
        + 1,  # Initial prompt + (each tool call = 2 messages)
    )
