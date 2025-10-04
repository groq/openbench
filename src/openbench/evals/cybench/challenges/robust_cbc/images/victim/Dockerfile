FROM ubuntu:focal

RUN apt-get update -y && \
    apt-get install -y xinetd netcat-openbsd socat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m user

COPY app /home/user/
COPY xinetd /etc/xinetd.d/user

WORKDIR /home/user

RUN chown -R root:root . && \
    chmod a+x server

EXPOSE 1337

CMD ["/usr/sbin/xinetd", "-dontfork"]
