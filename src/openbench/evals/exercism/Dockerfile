# Dockerfile
FROM debian:bookworm

# Base deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg git jq xz-utils bash \
    python3 python3-venv python3-pip python3-dev \
    golang-go default-jre cmake \
    xvfb xauth socat netcat-openbsd procps strace lsof \
    # VS Code/Electron runtime libs commonly needed headless:
    libx11-6 libx11-xcb1 libxkbfile1 libsecret-1-0 libnss3 libgbm1 \
    libasound2 libxshmfence1 libxext6 libxrender1 libxi6 libsm6 libice6 \
 && rm -rf /var/lib/apt/lists/*

# Desktop VS Code
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/packages.microsoft.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
      > /etc/apt/sources.list.d/vscode.list && \
    apt-get update && apt-get install -y code && rm -rf /var/lib/apt/lists/*

# Node 20 + pnpm + dotenvx
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs \
 && npm i -g pnpm@9 @dotenvx/dotenvx \
 && rm -rf /var/lib/apt/lists/*

 ARG ROO_EXT_VER=3.22.0

 # Clean target dirs to avoid leftover versions
 RUN rm -rf /opt/vscode-extensions && mkdir -p /opt/vscode-extensions \
     && rm -rf /opt/vscode-user && mkdir -p /opt/vscode-user
 
 # Uninstall any preinstalled Roo extension (no-op if absent)
 RUN xvfb-run -a code --uninstall-extension RooVeterinaryInc.roo-cline \
       --extensions-dir /opt/vscode-extensions \
       --user-data-dir /opt/vscode-user || true
 
 # Install the exact version
 RUN xvfb-run -a code --install-extension RooVeterinaryInc.roo-cline@${ROO_EXT_VER} \
       --extensions-dir /opt/vscode-extensions \
       --user-data-dir /opt/vscode-user

# Roo CLI  
RUN git clone https://github.com/cte/roo-cli /opt/roo-cli \
 && cd /opt/roo-cli && pnpm install --frozen-lockfile \
 && echo "# Default .env for roo-cli" > .env \
 && echo "ROO_CODE_IPC_SOCKET_PATH=/tmp/roo-code.sock" >> .env

# OpenCode CLI (global) -
RUN npm install -g opencode-ai

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Python virtual environment support for Aider (will be installed per-task)
RUN curl -LsSf https://aider.chat/install.sh | sh && \
    # Move aider to global path
    cp ~/.local/bin/aider /usr/local/bin/aider || \
    cp /root/.local/bin/aider /usr/local/bin/aider || \

# VS Code user settings
RUN mkdir -p /opt/vscode-user/User && \
    printf '{\n  "security.workspace.trust.enabled": false,\n  "telemetry.telemetryLevel": "off",\n  "extensions.autoUpdate": false,\n  "roo-cline.autoImportSettingsPath": "/etc/roo/roo-code-settings.json"\n}\n' \
      > /opt/vscode-user/User/settings.json

WORKDIR /workspace

# Create a startup script that launches headless VS Code and waits for the Roo socket
RUN cat > /usr/local/bin/start-vscode.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
 
# Ensure defaults
export ROO_CODE_IPC_SOCKET_PATH="${ROO_CODE_IPC_SOCKET_PATH:-/tmp/roo-code.sock}"
export VSCODE_EXT_DIR="${VSCODE_EXT_DIR:-/opt/vscode-extensions}"
export VSCODE_USER_DIR="${VSCODE_USER_DIR:-/opt/vscode-user}"

# Prepare Roo and VS Code settings
mkdir -p /etc/roo "$VSCODE_EXT_DIR" "$VSCODE_USER_DIR" /tmp
printf '{ "ipcSocketPath": "%s" }\n' "${ROO_CODE_IPC_SOCKET_PATH}" > /etc/roo/roo-code-settings.json

# Wait up to 60s for the Roo IPC socket
for i in $(seq 1 60); do
  if [ -S "${ROO_CODE_IPC_SOCKET_PATH}" ]; then
    echo "Roo socket is ready."
    break
  fi
  if [ $((i % 5)) -eq 0 ]; then
    echo "--- VS Code status (t=${i}s) ---"
    echo "Processes:"
    ps aux | grep -E "(code|Code)" | grep -v grep || echo "No code processes found"
    echo "Last 50 VS Code log lines:"
    tail -n 50 /tmp/code.log || true
    echo "Socket file:"
    ls -la "${ROO_CODE_IPC_SOCKET_PATH}" || echo "Socket not present yet"
    echo "Unix sockets (lsof):"
    lsof -U | grep -E "(roo-code|code|Code)" || true
    echo "Installed extensions (if available):"
    xvfb-run -a code --list-extensions --extensions-dir "${VSCODE_EXT_DIR}" --user-data-dir "${VSCODE_USER_DIR}" || true
  fi
  sleep 1
done

if [ ! -S "${ROO_CODE_IPC_SOCKET_PATH}" ]; then
  echo "ERROR: Roo socket not created after waiting. Recent VS Code logs:"
  tail -n 200 /tmp/code.log || true
  echo "Processes:"
  ps aux | grep -E "(code|Code)" | grep -v grep || true
  echo "Unix sockets (lsof):"
  lsof -U | grep -E "(roo-code|code|Code)" || true
else
  echo "Socket ready -> ${ROO_CODE_IPC_SOCKET_PATH}"
  ls -la "${ROO_CODE_IPC_SOCKET_PATH}" || true
  echo "Testing socket connectivity (nc -U)..."
  if timeout 2 bash -lc "echo '' | nc -U '${ROO_CODE_IPC_SOCKET_PATH}'" >/tmp/roo-sock-test.log 2>&1; then
    echo "Socket accepted test connection. Test output:"
    tail -n +1 /tmp/roo-sock-test.log || true
  else
    echo "Socket test failed (may be expected); test output:"
    cat /tmp/roo-sock-test.log || true
  fi
fi

# Continue with the container command to keep it alive
exec "$@"
EOF
RUN chmod +x /usr/local/bin/start-vscode.sh

# Use the startup script as entrypoint; default command keeps container alive
ENTRYPOINT ["/usr/local/bin/start-vscode.sh"]
CMD ["bash", "-lc", "tail -f /dev/null"]
