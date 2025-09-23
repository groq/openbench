FROM debian:bookworm

# Base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg git jq xz-utils bash python3 python3-pip python3-venv python3-dev golang-go default-jre default-jdk unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and pnpm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . ~/.cargo/env \
    && rustup default stable

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Gradle wrapper dependencies
RUN apt-get update && apt-get install -y unzip \
    && rm -rf /var/lib/apt/lists/*

# Add Rust and uv to PATH for all users
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Common setup
WORKDIR /workspace

# Agent-specific and run time installation is handled by DockerManager

# Default command
CMD ["tail", "-f", "/dev/null"]
