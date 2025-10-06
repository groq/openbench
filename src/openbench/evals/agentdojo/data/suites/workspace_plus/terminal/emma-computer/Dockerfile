# https://hub.docker.com/layers/library/ubuntu/24.04/images/sha256-6e75a10070b0fcb0bead763c5118a369bc7cc30dfc1b0749c491bbb21f15c3c7
FROM ubuntu:24.04@sha256:6e75a10070b0fcb0bead763c5118a369bc7cc30dfc1b0749c491bbb21f15c3c7

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install desktop environment and basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    dnsutils \
    git \
    iputils-ping \
    nano \
    net-tools \
    python3 \
    python3-pip \
    sudo \
    vim \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create script that sets up /etc/hosts
COPY <<'EOF' /usr/bin/host-setup.sh
#!/bin/bash
mock_ip=$(dig +short mock_gist_server)
echo "${mock_ip} gist.github.com" >> /etc/hosts
EOF
RUN chmod 700 /usr/bin/host-setup.sh

# Create the emma user
# Allow emma user to run host-setup.sh as root
RUN useradd -m -s /bin/bash emma && \
    echo "emma ALL=(ALL) NOPASSWD: /usr/bin/host-setup.sh" >> /etc/sudoers.d/emma

# Create entrypoint script
COPY <<'EOF' /usr/bin/entrypoint.sh
#!/bin/bash
sudo /usr/bin/host-setup.sh
sleep infinity
EOF
RUN chmod +x /usr/bin/entrypoint.sh
RUN chown emma:emma /usr/bin/entrypoint.sh

# Copy over certs for mock server
COPY mock-gist-server/certs/nginx.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Switch to the emma user
USER emma
WORKDIR /home/emma

# Copy the home directory files
COPY --chown=emma:emma emma-computer/home-dir/ /home/emma/

# Set the entrypoint
ENTRYPOINT ["/usr/bin/entrypoint.sh"]
