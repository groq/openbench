"""
Registry for remote MCP handlers.

Provides automatic provider detection and handler dispatch based on model name.
"""

from openbench.evals.remote_mcp.base import RemoteMCPHandler
from openbench.evals.remote_mcp.groq import GroqRemoteMCPHandler
from openbench.evals.remote_mcp.anthropic import AnthropicRemoteMCPHandler

HANDLERS: list[type[RemoteMCPHandler]] = [
    GroqRemoteMCPHandler,
    AnthropicRemoteMCPHandler,
]


def get_supported_providers() -> list[str]:
    """Return list of supported provider prefixes."""
    return ["groq/*", "anthropic/*"]


def get_remote_mcp_handler(
    model_name: str,
    tool_discovery: str | None = None,
) -> RemoteMCPHandler:
    """Get the appropriate remote MCP handler for the given model.

    Args:
        model_name: Full model name (e.g., "groq/llama-3.3-70b")
        tool_discovery: Optional tool discovery mode (provider-specific)

    Returns:
        Configured RemoteMCPHandler instance

    Raises:
        ValueError: If no handler supports the provider, or if tool_discovery
            is invalid for the provider
    """
    for handler_cls in HANDLERS:
        if handler_cls.supports_provider(model_name):
            valid_opts = handler_cls.valid_tool_discovery_options()

            if tool_discovery and tool_discovery not in valid_opts:
                provider_prefix = model_name.split("/")[0]
                raise ValueError(
                    f"tool_discovery='{tool_discovery}' is not valid for "
                    f"'{provider_prefix}' provider. "
                    f"Valid options: {valid_opts if valid_opts else 'none (omit parameter)'}"
                )

            return handler_cls(model_name, tool_discovery)

    provider_prefix = model_name.split("/")[0] if "/" in model_name else model_name
    supported = get_supported_providers()

    raise ValueError(
        f"No remote MCP handler available for provider '{provider_prefix}'. "
        f"The 'minimal-servers-remote' strategy requires server-side MCP support. "
        f"Supported providers: {', '.join(supported)}"
    )
