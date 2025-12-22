"""
Base class for remote MCP handlers.

Each provider (Groq, Anthropic, etc.) implements this interface to handle
server-side MCP execution with provider-specific APIs and features.
"""

from abc import ABC, abstractmethod

from inspect_ai.solver._task_state import TaskState


class RemoteMCPHandler(ABC):
    """Base class for provider-specific remote MCP implementations."""

    def __init__(self, model_name: str, tool_discovery: str | None = None):
        """Initialize the handler.

        Args:
            model_name: Full model name (e.g., "groq/llama-3.3-70b")
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
    def supports_provider(cls, model_name: str) -> bool:
        """Return True if this handler supports the given model.

        Args:
            model_name: Full model name to check

        Returns:
            True if this handler can process this model
        """
        pass

    @classmethod
    @abstractmethod
    def valid_tool_discovery_options(cls) -> list[str]:
        """Return list of valid tool_discovery values for this provider.

        Returns:
            List of valid option strings (empty list if none supported)
        """
        pass

    def get_model_id(self) -> str:
        """Extract the model ID from the full model name.

        E.g., "groq/llama-3.3-70b" -> "llama-3.3-70b"
        """
        parts = self.model_name.split("/", 1)
        return parts[1] if len(parts) > 1 else self.model_name
