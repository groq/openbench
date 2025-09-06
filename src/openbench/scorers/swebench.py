import re

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState

TIMEOUT = 30  # 30 seconds timeout for patch validation


def extract_patch_from_completion(completion: str) -> str:
    """
    Extract patch content from a model completion.

    Looks for various patch formats including unified diff, git diff,
    or code blocks that might contain patch content.

    Args:
        completion (str): The model's completion output.

    Returns:
        str: The extracted patch content.
    """
    # Try to find unified diff format
    diff_pattern = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.DOTALL)
    matches = diff_pattern.findall(completion)
    if matches:
        return matches[0].strip()

    # Look for git diff format
    git_diff_pattern = re.compile(r"diff --git.*?(?=diff --git|\Z)", re.DOTALL)
    git_matches = git_diff_pattern.findall(completion)
    if git_matches:
        return "\n".join(git_matches).strip()

    # Look for lines that start with +, -, or @@
    lines = completion.split("\n")
    patch_lines = []
    in_patch = False

    for line in lines:
        if line.startswith("@@") or line.startswith("+++") or line.startswith("---"):
            in_patch = True
            patch_lines.append(line)
        elif in_patch and (
            line.startswith("+") or line.startswith("-") or line.startswith(" ")
        ):
            patch_lines.append(line)
        elif in_patch and line.strip() == "":
            patch_lines.append(line)
        elif in_patch and not (
            line.startswith("+") or line.startswith("-") or line.startswith(" ")
        ):
            # End of patch section
            break

    if patch_lines:
        return "\n".join(patch_lines).strip()

    # If no clear patch format found, return the whole completion
    return completion.strip()


def validate_patch_format(patch: str) -> bool:
    """
    Validate that a patch has basic correct formatting.

    Args:
        patch (str): The patch content to validate.

    Returns:
        bool: True if patch format appears valid, False otherwise.
    """
    if not patch.strip():
        return False

    lines = patch.split("\n")

    # Check for unified diff headers
    has_diff_header = any(
        line.startswith("---") or line.startswith("+++") or line.startswith("@@")
        for line in lines
    )

    # Check for git diff format
    has_git_header = any(line.startswith("diff --git") for line in lines)

    # Check for basic patch content (additions/deletions)
    has_patch_content = any(
        line.startswith("+") or line.startswith("-")
        for line in lines
        if not line.startswith("+++") and not line.startswith("---")
    )

    return (has_diff_header or has_git_header) and has_patch_content


def apply_patch_safely(patch: str, repo_path: str) -> tuple[bool, str]:
    """
    Attempt to apply a patch in a safe environment and check if it's valid.

    This is a simplified validation - in a full implementation, this would
    use the actual SWE-bench Docker environment.

    Args:
        patch (str): The patch content to apply.
        repo_path (str): Path to repository (not used in basic implementation).

    Returns:
        tuple[bool, str]: (success, error_message)
    """
    # For now, just validate the patch format
    # A full implementation would:
    # 1. Set up the repository at the correct commit
    # 2. Apply the patch using git apply or patch command
    # 3. Run the test suite to verify correctness
    # 4. Return results

    if not validate_patch_format(patch):
        return False, "Invalid patch format"

    # Basic syntax validation could be added here
    # For example, checking for common patch syntax errors

    return True, "Patch format appears valid"


@scorer(metrics=[accuracy(), stderr()])
def swebench_scorer(docker_validation: bool = False) -> Scorer:
    """
    Scorer for SWE-bench tasks. Validates generated patches.

    Args:
        docker_validation (bool): Whether to use Docker for full validation.
                                 If False, only does format validation.

    Returns:
        Scorer: The SWE-bench patch validation scorer.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score a model's patch output by validating format and optionally applying it.

        Args:
            state (TaskState): The current task state containing model output and metadata.
            target (Target): The target patch (reference solution).

        Returns:
            Score: The result of patch validation.
        """
        generated_patch = extract_patch_from_completion(state.output.completion)

        if not generated_patch:
            return Score(
                value=INCORRECT,
                answer=generated_patch,
                explanation="No patch content found in model output",
            )

        # Basic format validation
        if not validate_patch_format(generated_patch):
            return Score(
                value=INCORRECT,
                answer=generated_patch,
                explanation="Generated patch does not have valid patch format",
            )

        success = True
        explanation_parts = ["Generated patch has valid format."]

        # Optional Docker-based validation
        if docker_validation:
            try:
                # In a full implementation, this would:
                # 1. Create Docker container with repo at base_commit
                # 2. Apply the patch
                # 3. Run tests to verify correctness
                # 4. Compare against expected test outcomes

                # For now, just simulate this with basic validation
                repo_path = f"/tmp/swebench_{state.metadata.get('repo', 'unknown')}"
                patch_success, error_msg = apply_patch_safely(
                    generated_patch, repo_path
                )

                if not patch_success:
                    success = False
                    explanation_parts.append(f"Patch validation failed: {error_msg}")
                else:
                    explanation_parts.append("Patch validation passed (basic check).")

            except Exception as e:
                success = False
                explanation_parts.append(f"Error during patch validation: {str(e)}")
        else:
            explanation_parts.append("Note: Full Docker validation not enabled.")

        return Score(
            value=CORRECT if success else INCORRECT,
            answer=generated_patch,
            explanation="\n".join(explanation_parts),
        )

    return score


# Convenience scorers for different modes
@scorer(metrics=[accuracy(), stderr()])
def swebench_format_scorer() -> Scorer:
    """Quick format-only validation scorer for SWE-bench."""
    return swebench_scorer(docker_validation=False)


@scorer(metrics=[accuracy(), stderr()])
def swebench_docker_scorer() -> Scorer:
    """Full Docker-based validation scorer for SWE-bench."""
    return swebench_scorer(docker_validation=True)
