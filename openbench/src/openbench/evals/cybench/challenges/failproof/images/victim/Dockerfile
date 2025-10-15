FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install -y lib32z1 xinetd socat && \
    pip3 install requests && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m user && \
    chown -R root:root /home/user

COPY app /home/user/
COPY xinetd /etc/xinetd.d/user

WORKDIR /home/user

EXPOSE 1337

CMD ["/usr/sbin/xinetd", "-dontfork"]
