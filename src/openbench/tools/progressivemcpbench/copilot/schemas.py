"""
Schemas for Copilot server and connection configuration.

Credit: Adapted from LiveMCPBench baseline schemas:
https://github.com/icip-cas/LiveMCPBench/blob/main/baseline/mcp_copilot/schemas.py
"""

from typing import Any

import mcp.types as types
from pydantic import BaseModel, model_validator


class ServerConfig(BaseModel):
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, Any] = {}

    @model_validator(mode="after")
    def check_command_or_url(self):
        if not self.command and not self.url:
            raise ValueError("Either 'command' or 'url' must be provided")
        return self


class Server(BaseModel):
    """Definition for a server the client can connect."""

    name: str
    """The name of the server."""
    description: str | None = None
    """A human-readable description of the server."""
    config: ServerConfig
    """The configuration for the server."""

    tools: list[types.Tool] | None = None
    """The tools available on the server."""
