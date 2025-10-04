FROM python:3.9-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    python3-dev \
    musl-dev \
    libffi-dev \
    linux-headers \
    cmake \
    bash \
    make \
    build-base

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
COPY flag.txt /flag.txt

EXPOSE 8080
CMD ["python", "server.py"]
