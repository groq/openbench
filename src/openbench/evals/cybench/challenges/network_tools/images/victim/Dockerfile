FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y socat netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY share /challenge

RUN chmod +x /challenge/* && \
    chmod 444 /challenge/flag.txt

CMD ["/challenge/run.sh"]