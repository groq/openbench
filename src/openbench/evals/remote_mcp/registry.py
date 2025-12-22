"""
Registry for remote MCP handlers.

Provides automatic provider detection and handler dispatch based on ModelAPI type.
"""

from typing import TYPE_CHECKING

from openbench.evals.remote_mcp.base import RemoteMCPHandler
from openbench.evals.remote_mcp.groq import GroqRemoteMCPHandler
from openbench.evals.remote_mcp.anthropic import AnthropicRemoteMCPHandler

if TYPE_CHECKING:
    from inspect_ai.model._model import ModelAPI

HANDLERS: list[type[RemoteMCPHandler]] = [
    GroqRemoteMCPHandler,
    AnthropicRemoteMCPHandler,
]


def get_supported_providers() -> list[str]:
    """Return list of supported provider display names."""
    return [h.provider_name() for h in HANDLERS]


def get_remote_mcp_handler(
    api: "ModelAPI",
    tool_discovery: str | None = None,
) -> RemoteMCPHandler:
    """Get the appropriate remote MCP handler for the given ModelAPI.

    Args:
        api: The ModelAPI instance from get_model().api
        tool_discovery: Optional tool discovery mode (provider-specific)

    Returns:
        Configured RemoteMCPHandler instance

    Raises:
        ValueError: If no handler supports the API type, or if tool_discovery
            is invalid for the provider
    """
    for handler_cls in HANDLERS:
        if handler_cls.supports_api(api):
            valid_opts = handler_cls.valid_tool_discovery_options()

            if tool_discovery and tool_discovery not in valid_opts:
                raise ValueError(
                    f"tool_discovery='{tool_discovery}' is not valid for "
                    f"'{handler_cls.provider_name()}' provider. "
                    f"Valid options: {valid_opts if valid_opts else 'none (omit parameter)'}"
                )

            return handler_cls(api.model_name, tool_discovery)

    supported = get_supported_providers()

    raise ValueError(
        f"No remote MCP handler available for this model API. "
        f"The 'minimal-servers-remote' strategy requires server-side MCP support. "
        f"Supported providers: {', '.join(supported)}"
    )
