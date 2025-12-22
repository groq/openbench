"""
Base class for remote MCP handlers.

Each provider (Groq, Anthropic, etc.) implements this interface to handle
server-side MCP execution with provider-specific APIs and features.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from inspect_ai.solver._task_state import TaskState

if TYPE_CHECKING:
    from inspect_ai.model._model import ModelAPI


class RemoteMCPHandler(ABC):
    """Base class for provider-specific remote MCP implementations."""

    def __init__(self, model_name: str, tool_discovery: str | None = None):
        """Initialize the handler.

        Args:
            model_name: Model name (e.g., "claude-sonnet-4-5-20250929")
            tool_discovery: Optional tool discovery mode (provider-specific)
        """
        self.model_name = model_name
        self.tool_discovery = tool_discovery

    @abstractmethod
    async def execute(
        self,
        state: TaskState,
        required_servers: list[str],
        servers_config: dict,
        system_message: str,
    ) -> TaskState:
        """Execute the remote MCP call and return updated state.

        Args:
            state: Current task state with input_text
            required_servers: List of MCP server names needed for this task
            servers_config: Server configuration dict with descriptions
            system_message: System prompt to use

        Returns:
            Updated TaskState with completion and metadata
        """
        pass

    @classmethod
    @abstractmethod
    def supports_api(cls, api: "ModelAPI") -> bool:
        """Return True if this handler supports the given ModelAPI.

        Args:
            api: The ModelAPI instance to check

        Returns:
            True if this handler can process this API type
        """
        pass

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str:
        """Return the display name of this provider (e.g., 'groq', 'anthropic')."""
        pass

    @classmethod
    @abstractmethod
    def valid_tool_discovery_options(cls) -> list[str]:
        """Return list of valid tool_discovery values for this provider.

        Returns:
            List of valid option strings (empty list if none supported)
        """
        pass
