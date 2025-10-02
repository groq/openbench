FROM python:3.9-slim-buster

RUN apt-get update -y && \
    apt-get install -y xinetd socat && \
    pip3 install networkx pycryptodome && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m user && \
    chown -R root:root /home/user

COPY app /home/user/
COPY xinetd /etc/xinetd.d/user

WORKDIR /home/user

EXPOSE 9999

CMD ["/usr/sbin/xinetd", "-dontfork"]
