"""
SWE-Agent tools integration for Inspect AI.
Provides bridge between SWE-Agent tools and Inspect's tool framework.
"""

import json
import logging
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from inspect_ai.tool import ToolDef, ToolParams, ToolParam
from inspect_ai.util import sandbox, store_as
from inspect_ai.util._store_model import StoreModel
from pydantic import Field

logger = logging.getLogger(__name__)

# Keep in sync with SWE-Agent version
SWE_AGENT_URL = "https://github.com/SWE-agent/SWE-agent.git"
SWE_AGENT_BRANCH = "v1.0.1"


class SandboxStoreState(StoreModel):
    """State for sandbox environment used by SWE-Agent tools."""

    current_file: Optional[str] = Field(default=None)
    first_line: int = Field(default=0)


async def setup_sweagent_environment(window: int = 100, overlap: int = 2) -> None:
    """
    Setup the SWE-Agent environment in the sandbox.

    Args:
        window: Size of the file viewing window (lines).
        overlap: Number of lines to overlap when scrolling.
    """
    # Initialize registry file with default values
    registry_content = {
        "WINDOW": str(window),
        "OVERLAP": str(overlap),
    }
    await sandbox().write_file("/root/.swe-agent-env", json.dumps(registry_content))


async def sweagent_tooldefs(
    include_bundles: List[str],
) -> Dict[str, List[ToolDef]]:
    """
    Convert SWE-Agent tools to Inspect ToolDefs.

    This is a simplified version that creates the essential tools
    without requiring the full SWE-Agent package installation.

    Args:
        include_bundles: List of bundle names to include.

    Returns:
        Dictionary mapping bundle names to lists of ToolDef objects.
    """
    tools: Dict[str, List[ToolDef]] = {}

    # Define tools for each bundle
    if "defaults" in include_bundles:
        tools["defaults"] = create_default_tools()

    if "search" in include_bundles:
        tools["search"] = create_search_tools()

    if "edit_replace" in include_bundles:
        tools["edit_replace"] = create_edit_tools()

    return tools


def create_default_tools() -> List[ToolDef]:
    """Create default file navigation tools."""
    tools = []

    # goto tool
    async def goto(line_number: int) -> str:
        """Go to a specific line in the current file."""
        state = store_as(SandboxStoreState)
        if not state.current_file:
            return "No file is currently open. Use 'open' to open a file first."

        state.first_line = max(1, line_number)

        # Update registry
        registry = {"CURRENT_FILE": state.current_file, "FIRST_LINE": state.first_line}
        await sandbox().write_file("/root/.swe-agent-env", json.dumps(registry))

        # Display the file from the new position
        result = await sandbox().exec(
            [
                "bash",
                "-c",
                f"sed -n '{state.first_line},{state.first_line + 99}p' {shlex.quote(state.current_file)}",
            ]
        )
        return result.stdout

    tools.append(
        ToolDef(
            tool=goto,
            name="goto",
            description="Go to a specific line number in the current file",
            parameters=ToolParams(
                properties={
                    "line_number": ToolParam(
                        type="number", description="Line number to go to"
                    )
                },
                required=["line_number"],
            ),
        )
    )

    # open tool
    async def open_file(file_path: str, line_number: Optional[int] = None) -> str:
        """Open a file and optionally go to a specific line."""
        state = store_as(SandboxStoreState)

        # Check if file exists
        check_result = await sandbox().exec(["test", "-f", file_path])
        if not check_result.success:
            return f"File {file_path} not found"

        state.current_file = file_path
        state.first_line = line_number or 1

        # Update registry
        registry = {"CURRENT_FILE": state.current_file, "FIRST_LINE": state.first_line}
        await sandbox().write_file("/root/.swe-agent-env", json.dumps(registry))

        # Display the file
        result = await sandbox().exec(
            [
                "bash",
                "-c",
                f"sed -n '{state.first_line},{state.first_line + 99}p' {shlex.quote(file_path)}",
            ]
        )
        return f"Opened {file_path} at line {state.first_line}:\n{result.stdout}"

    tools.append(
        ToolDef(
            tool=open_file,
            name="open",
            description="Open a file and display its contents",
            parameters=ToolParams(
                properties={
                    "file_path": ToolParam(
                        type="string", description="Path to the file to open"
                    ),
                    "line_number": ToolParam(
                        type="number", description="Optional line number to start at"
                    ),
                },
                required=["file_path"],
            ),
        )
    )

    # create tool
    async def create(file_path: str) -> str:
        """Create a new file."""
        # Ensure directory exists
        dir_path = Path(file_path).parent
        await sandbox().exec(["mkdir", "-p", str(dir_path)])

        # Create empty file
        result = await sandbox().exec(["touch", file_path])
        if result.success:
            state = store_as(SandboxStoreState)
            state.current_file = file_path
            state.first_line = 1
            return f"Created file: {file_path}"
        return f"Failed to create file: {file_path}"

    tools.append(
        ToolDef(
            tool=create,
            name="create",
            description="Create a new file",
            parameters=ToolParams(
                properties={
                    "file_path": ToolParam(
                        type="string", description="Path for the new file"
                    )
                },
                required=["file_path"],
            ),
        )
    )

    # scroll_down tool
    async def scroll_down() -> str:
        """Scroll down in the current file."""
        state = store_as(SandboxStoreState)
        if not state.current_file:
            return "No file is currently open"

        state.first_line += 98  # 100 - 2 overlap

        result = await sandbox().exec(
            [
                "bash",
                "-c",
                f"sed -n '{state.first_line},{state.first_line + 99}p' {shlex.quote(state.current_file)}",
            ]
        )

        if not result.stdout:
            state.first_line -= 98  # Revert if at end
            return "Already at end of file"

        return result.stdout

    tools.append(
        ToolDef(
            tool=scroll_down,
            name="scroll_down",
            description="Scroll down in the current file",
            parameters=ToolParams(),
        )
    )

    # scroll_up tool
    async def scroll_up() -> str:
        """Scroll up in the current file."""
        state = store_as(SandboxStoreState)
        if not state.current_file:
            return "No file is currently open"

        state.first_line = max(1, state.first_line - 98)

        result = await sandbox().exec(
            [
                "bash",
                "-c",
                f"sed -n '{state.first_line},{state.first_line + 99}p' {shlex.quote(state.current_file)}",
            ]
        )
        return result.stdout

    tools.append(
        ToolDef(
            tool=scroll_up,
            name="scroll_up",
            description="Scroll up in the current file",
            parameters=ToolParams(),
        )
    )

    return tools


def create_search_tools() -> List[ToolDef]:
    """Create file search tools."""
    tools = []

    # find_file tool
    async def find_file(file_name: str, directory: Optional[str] = None) -> str:
        """Find files by name pattern."""
        search_dir = directory or "/testbed"
        result = await sandbox().exec(
            ["find", search_dir, "-name", file_name, "-type", "f"], timeout=30
        )

        if not result.stdout:
            return f"No files found matching '{file_name}'"

        return f"Files found:\n{result.stdout}"

    tools.append(
        ToolDef(
            tool=find_file,
            name="find_file",
            description="Find files by name pattern",
            parameters=ToolParams(
                properties={
                    "file_name": ToolParam(
                        type="string", description="File name or pattern to search for"
                    ),
                    "directory": ToolParam(
                        type="string",
                        description="Directory to search in (default: /testbed)",
                    ),
                },
                required=["file_name"],
            ),
        )
    )

    # search_dir tool
    async def search_dir(search_term: str, directory: Optional[str] = None) -> str:
        """Search for a term in files within a directory."""
        search_dir = directory or "/testbed"
        result = await sandbox().exec(
            ["grep", "-r", "-n", "--max-count=5", search_term, search_dir], timeout=30
        )

        if not result.stdout:
            return f"No occurrences of '{search_term}' found"

        # Limit output
        lines = result.stdout.split("\n")[:50]
        output = "\n".join(lines)
        if len(lines) == 50:
            output += "\n... (results truncated)"

        return f"Search results for '{search_term}':\n{output}"

    tools.append(
        ToolDef(
            tool=search_dir,
            name="search_dir",
            description="Search for a term in files within a directory",
            parameters=ToolParams(
                properties={
                    "search_term": ToolParam(
                        type="string", description="Term to search for"
                    ),
                    "directory": ToolParam(
                        type="string",
                        description="Directory to search in (default: /testbed)",
                    ),
                },
                required=["search_term"],
            ),
        )
    )

    # search_file tool
    async def search_file(search_term: str, file_path: Optional[str] = None) -> str:
        """Search for a term in the current file or a specific file."""
        state = store_as(SandboxStoreState)
        target_file = file_path or state.current_file

        if not target_file:
            return "No file specified and no file is currently open"

        result = await sandbox().exec(["grep", "-n", search_term, target_file])

        if not result.stdout:
            return f"No occurrences of '{search_term}' found in {target_file}"

        # Limit output
        lines = result.stdout.split("\n")[:20]
        output = "\n".join(lines)
        if len(lines) == 20:
            output += "\n... (results truncated)"

        return f"Search results in {target_file}:\n{output}"

    tools.append(
        ToolDef(
            tool=search_file,
            name="search_file",
            description="Search for a term in a file",
            parameters=ToolParams(
                properties={
                    "search_term": ToolParam(
                        type="string", description="Term to search for"
                    ),
                    "file_path": ToolParam(
                        type="string",
                        description="File to search in (default: current file)",
                    ),
                },
                required=["search_term"],
            ),
        )
    )

    return tools


def create_edit_tools() -> List[ToolDef]:
    """Create file editing tools."""
    tools = []

    # str_replace_based_edit_tool
    async def str_replace_based_edit_tool(
        old_str: str,
        new_str: str,
        file_path: Optional[str] = None,
        replace_all: bool = False,
    ) -> str:
        """
        Replace occurrences of old_str with new_str in a file.
        """
        state = store_as(SandboxStoreState)
        target_file = file_path or state.current_file

        if not target_file:
            return "No file specified and no file is currently open"

        # Read the file
        content_result = await sandbox().exec(["cat", target_file])
        if not content_result.success:
            return f"Failed to read file: {target_file}"

        content = content_result.stdout

        # Check if old_str exists
        if old_str not in content:
            return f"The string '{old_str[:50]}...' was not found in {target_file}"

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_str, new_str)
            count = content.count(old_str)
            message = f"Replaced {count} occurrences"
        else:
            new_content = content.replace(old_str, new_str, 1)
            message = "Replaced 1 occurrence"

        # Write back to file
        await sandbox().write_file(target_file, new_content)

        # Show a preview of the change
        preview_lines = 5
        changed_lines = []
        for i, (old_line, new_line) in enumerate(
            zip(content.split("\n"), new_content.split("\n"))
        ):
            if old_line != new_line:
                start = max(0, i - 2)
                end = min(len(new_content.split("\n")), i + 3)
                preview = "\n".join(new_content.split("\n")[start:end])
                changed_lines.append(f"Around line {i + 1}:\n{preview}")
                if len(changed_lines) >= preview_lines:
                    break

        preview_text = "\n\n".join(changed_lines[:preview_lines])
        if len(changed_lines) > preview_lines:
            preview_text += (
                f"\n\n... and {len(changed_lines) - preview_lines} more changes"
            )

        return f"{message} in {target_file}\n\nPreview of changes:\n{preview_text}"

    tools.append(
        ToolDef(
            tool=str_replace_based_edit_tool,
            name="str_replace_based_edit_tool",
            description="Replace string in file with exact string matching",
            parameters=ToolParams(
                properties={
                    "old_str": ToolParam(
                        type="string", description="Exact string to replace"
                    ),
                    "new_str": ToolParam(
                        type="string", description="New string to replace with"
                    ),
                    "file_path": ToolParam(
                        type="string",
                        description="File to edit (default: current file)",
                    ),
                    "replace_all": ToolParam(
                        type="boolean",
                        description="Replace all occurrences (default: False)",
                    ),
                },
                required=["old_str", "new_str"],
            ),
        )
    )

    return tools


def generate_dockerfile_content(image_name: str, bundles: List[str]) -> str:
    """
    Generate Dockerfile content for SWE-Agent environment.

    Args:
        image_name: Base Docker image name.
        bundles: List of SWE-Agent bundles to include.

    Returns:
        String containing Dockerfile content.
    """
    dockerfile_content = f"""# Stage 1: Clone SWE-Agent repository
FROM alpine/git as swe-agent-source
RUN git clone --depth 1 --branch {SWE_AGENT_BRANCH} {SWE_AGENT_URL} /swe-agent

# Stage 2: Build final image
FROM {image_name}

# Install packages required by SWE-Agent tools
RUN --mount=type=cache,target=/root/.cache/pip \\
    pip install flake8

# Create directory structure for SWE-Agent tools
RUN mkdir -p /root/tools
"""

    # Copy each specified bundle
    for bundle in bundles:
        dockerfile_content += f"""
# Copy {bundle} bundle
COPY --from=swe-agent-source /swe-agent/tools/{bundle} /root/tools/{bundle}
"""

    # Setup registry environment
    dockerfile_content += """
# Initialize registry file
RUN echo "{}" > /root/.swe-agent-env

# Make executable files in bin directories executable
RUN find /root/tools -path "*/bin/*" -type f -exec chmod +x {} \\;
"""

    return dockerfile_content
