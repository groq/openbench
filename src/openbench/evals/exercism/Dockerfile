FROM debian:bookworm

# Base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg git jq xz-utils bash python3 python3-pip python3-venv python3-dev golang-go default-jre default-jdk unzip xvfb xauth socat netcat-openbsd procps strace lsof libx11-6 libx11-xcb1 libxkbfile1 libsecret-1-0 libnss3 libgbm1 libasound2 libxshmfence1 libxext6 libxrender1 libxi6 libsm6 libice6 \
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

# Agent-specific installation
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/packages.microsoft.gpg
RUN ARCH=$(dpkg --print-architecture) && echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list
RUN apt-get update && apt-get install -y code && rm -rf /var/lib/apt/lists/*
RUN npm i -g @dotenvx/dotenvx
RUN rm -rf /opt/vscode-extensions && mkdir -p /opt/vscode-extensions
RUN rm -rf /opt/vscode-user && mkdir -p /opt/vscode-user
RUN xvfb-run -a code --install-extension RooVeterinaryInc.roo-cline@3.22.0 --extensions-dir /opt/vscode-extensions --user-data-dir /opt/vscode-user
RUN git clone https://github.com/cte/roo-cli /opt/roo-cli
RUN cd /opt/roo-cli && pnpm install
RUN npm install -g tsx
RUN echo '# Default .env for roo-cli' > /opt/roo-cli/.env
RUN echo 'ROO_CODE_IPC_SOCKET_PATH=/tmp/roo-code.sock' >> /opt/roo-cli/.env
RUN mkdir -p /opt/vscode-user/User
RUN printf '{"security.workspace.trust.enabled": false,"telemetry.telemetryLevel": "off","extensions.autoUpdate": false,"roo-cline.autoImportSettingsPath": "/etc/roo/roo-code-settings.json"}\n' > /opt/vscode-user/User/settings.json

# Runtime setup
RUN mkdir -p /etc/roo /tmp
RUN printf '{ "ipcSocketPath": "/tmp/roo-code.sock" }\n' > /etc/roo/roo-code-settings.json

# Default command
CMD ["tail", "-f", "/dev/null"]
