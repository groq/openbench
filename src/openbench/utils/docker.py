"""
Docker command utilities for openbench.

This module contains all Docker-related command strings and installation
scripts used across agents and the Docker manager.
"""

from typing import List


# Base Dockerfile template for all agents
BASE_DOCKERFILE_TEMPLATE = """FROM debian:bookworm

# Base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    {base_packages} \\
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and pnpm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \\
    && apt-get install -y nodejs \\
    && npm install -g pnpm \\
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \\
    && . ~/.cargo/env \\
    && rustup default stable

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Gradle wrapper dependencies
RUN apt-get update && apt-get install -y unzip \\
    && rm -rf /var/lib/apt/lists/*

# Add Rust and uv to PATH for all users
ENV PATH="/root/.cargo/bin:/root/.local/bin:${{PATH}}"

# Common setup
WORKDIR /workspace

# Agent-specific installation
{agent_commands}

# Runtime setup
{runtime_setup}

# Default command
CMD ["tail", "-f", "/dev/null"]
"""


# Common base packages needed for all containers
COMMON_BASE_PACKAGES = [
    "ca-certificates",
    "curl",
    "wget",
    "gnupg",
    "git",
    "jq",
    "xz-utils",
    "bash",
]

# Language runtime packages for exercism support
LANGUAGE_RUNTIME_PACKAGES = [
    "python3",
    "python3-pip",
    "python3-venv",
    "python3-dev",
    "golang-go",
    "default-jre",
    "default-jdk",
    "unzip",  # For Gradle wrapper
]


# ============================================================================
# Agent-specific Docker commands
# ============================================================================


class AiderCommands:
    """Docker commands for Aider agent installation."""

    DOCKERFILE_COMMANDS = [
        "RUN curl -LsSf https://aider.chat/install.sh | sh",
        "RUN cp ~/.local/bin/aider /usr/local/bin/aider || cp /root/.local/bin/aider /usr/local/bin/aider",
    ]

    BASE_PACKAGES = [
        "python3",
        "python3-pip",
        "python3-venv",
        "git",  # Aider requires git
    ]


class OpenCodeCommands:
    """Docker commands for OpenCode agent installation."""

    DOCKERFILE_COMMANDS = ["RUN npm install -g opencode-ai"]

    BASE_PACKAGES = ["curl", "gnupg", "ca-certificates"]


class ClaudeCommands:
    """Docker commands for Claude agent installation."""

    DOCKERFILE_COMMANDS = ["RUN npm install -g @anthropic-ai/claude-code"]

    BASE_PACKAGES = ["curl", "gnupg", "ca-certificates"]


class RooCommands:
    """Docker commands for Roo agent installation."""

    @staticmethod
    def get_dockerfile_commands() -> List[str]:
        """Get Dockerfile commands for Roo installation."""
        return [
            # Install VS Code
            "RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/packages.microsoft.gpg",
            'RUN ARCH=$(dpkg --print-architecture) && echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list',
            "RUN apt-get update && apt-get install -y code && rm -rf /var/lib/apt/lists/*",
            # Install dotenvx (Node.js and pnpm already installed)
            "RUN npm i -g @dotenvx/dotenvx",
            # Set up VS Code directories
            "RUN rm -rf /opt/vscode-extensions && mkdir -p /opt/vscode-extensions",
            "RUN rm -rf /opt/vscode-user && mkdir -p /opt/vscode-user",
            # Install Roo extension
            "RUN xvfb-run -a code --install-extension RooVeterinaryInc.roo-cline@3.22.0 --extensions-dir /opt/vscode-extensions --user-data-dir /opt/vscode-user",
            # Clone roo-cli
            "RUN git clone https://github.com/cte/roo-cli /opt/roo-cli",
            # Install roo-cli dependencies
            "RUN cd /opt/roo-cli && pnpm install",
            # Install tsx globally for TypeScript execution
            "RUN npm install -g tsx",
            "RUN echo '# Default .env for roo-cli' > /opt/roo-cli/.env",
            "RUN echo 'ROO_CODE_IPC_SOCKET_PATH=/tmp/roo-code.sock' >> /opt/roo-cli/.env",
            # Set up VS Code user settings
            "RUN mkdir -p /opt/vscode-user/User",
            'RUN printf \'{"security.workspace.trust.enabled": false,"telemetry.telemetryLevel": "off","extensions.autoUpdate": false,"roo-cline.autoImportSettingsPath": "/etc/roo/roo-code-settings.json"}\\n\' > /opt/vscode-user/User/settings.json',
        ]

    BASE_PACKAGES = [
        # VS Code dependencies
        "wget",
        "gnupg",
        "ca-certificates",
        # X11 and display for headless VS Code
        "xvfb",
        "xauth",
        "socat",
        "netcat-openbsd",
        "procps",
        "strace",
        "lsof",
        # VS Code/Electron runtime libs
        "libx11-6",
        "libx11-xcb1",
        "libxkbfile1",
        "libsecret-1-0",
        "libnss3",
        "libgbm1",
        "libasound2",
        "libxshmfence1",
        "libxext6",
        "libxrender1",
        "libxi6",
        "libsm6",
        "libice6",
        # Git for roo-cli
        "git",
    ]

    VOLUMES = {
        "vscode_exts": "/opt/vscode-extensions",
        "vscode_user": "/opt/vscode-user",
        "node_cache": "/root/.npm",
    }

    RUNTIME_COMMANDS = [
        "mkdir -p /etc/roo /tmp",
        'printf \'{ "ipcSocketPath": "/tmp/roo-code.sock" }\\n\' > /etc/roo/roo-code-settings.json',
    ]

    ENV_REQUIREMENTS = [
        "ROO_CODE_IPC_SOCKET_PATH",
        "OPENROUTER_API_KEY",  # Roo commonly uses OpenRouter
    ]


# ============================================================================
# Docker Compose templates
# ============================================================================

EVAL_COMPOSE_TEMPLATE = """services:
  default:
    build: 
      context: .
      tags:
        - {container_tag}
    init: true
    command: tail -f /dev/null
    networks: [eval_net]
{environment_section}
{volumes_section}

networks:
  eval_net: {{ driver: bridge, internal: false }}

{volume_definitions}
"""

STANDARD_COMPOSE_TEMPLATE = """services:
  default:
    image: {container_tag}
    init: true
    command: tail -f /dev/null
    networks: [eval_net]
{environment_section}
{volumes_section}

networks:
  eval_net: {{ driver: bridge, internal: false }}

{volume_definitions}
"""


# ============================================================================
# Helper functions
# ============================================================================


def get_agent_docker_commands(agent_name: str) -> List[str]:
    """Get Docker commands for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of Docker RUN commands

    Raises:
        ValueError: If agent is not supported
    """
    commands_map = {
        "aider": AiderCommands.DOCKERFILE_COMMANDS,
        "opencode": OpenCodeCommands.DOCKERFILE_COMMANDS,
        "claude": ClaudeCommands.DOCKERFILE_COMMANDS,
        "roo": RooCommands.get_dockerfile_commands(),
    }

    if agent_name not in commands_map:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return commands_map[agent_name]


def get_agent_base_packages(agent_name: str) -> List[str]:
    """Get base packages required by a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of apt package names

    Raises:
        ValueError: If agent is not supported
    """
    packages_map = {
        "aider": AiderCommands.BASE_PACKAGES,
        "opencode": OpenCodeCommands.BASE_PACKAGES,
        "claude": ClaudeCommands.BASE_PACKAGES,
        "roo": RooCommands.BASE_PACKAGES,
    }

    if agent_name not in packages_map:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return packages_map[agent_name]
