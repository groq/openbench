# This defines a docker image which the agent can run commands in.
FROM kalilinux/kali-last-release@sha256:99d99f1c1349a5e1fa8f862875ad204ffc009599e2a6c4ed617d16f6d5bae241

# Avoid prompts during package installation.
ARG DEBIAN_FRONTEND=noninteractive

# Install 2025-04 Kali Linux archive signing key https://www.kali.org/blog/new-kali-archive-signing-key/
COPY ./archive-keyring.gpg /usr/share/keyrings/kali-archive-keyring.gpg

RUN apt-get update \
    && apt-get install -y \
    kali-linux-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/

# Keep the container running indefinitely.
CMD ["tail", "-f", "/dev/null"]
