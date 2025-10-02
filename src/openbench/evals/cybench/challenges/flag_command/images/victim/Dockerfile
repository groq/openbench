FROM python:3.11-alpine

# Install packages
RUN apk add --no-cache --update supervisor

# Upgrade pip
RUN python -m pip install --upgrade pip

# Add curl for health check.
RUN apk add --no-cache curl

# Install dependencies
RUN pip install Flask colorama

# Setup app
RUN mkdir -p /app

# Switch working environment
WORKDIR /app

# Copy flag
COPY flag.txt /flag.txt

# Add application
COPY challenge .

# Setup supervisor
COPY config/supervisord.conf /etc/supervisord.conf

# Expose port the server is reachable on
EXPOSE 1337

# Disable pycache
ENV PYTHONDONTWRITEBYTECODE=1

# start supervisord
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]