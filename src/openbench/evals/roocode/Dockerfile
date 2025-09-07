# Dockerfile
FROM python:3.11-slim

# system deps incl. bash + TLS certs + git for HF/GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git git-lfs curl build-essential && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# python deps used by your solver/setup steps
RUN pip install --no-cache-dir \
    huggingface_hub \
    datasets \
    requests

# keep container alive (Inspect will exec into it)
CMD ["tail", "-f", "/dev/null"]
# Create working directory
WORKDIR /workspace