FROM alpine:edge

RUN apk update && apk upgrade && apk add --no-cache \
    supervisor \
    pcre \
    pcre-dev \
    make \
    gcc \
    musl-dev \
    linux-headers \
    python3-dev \
    py3-pip \
    libffi-dev \
    netcat-openbsd \
    socat \
    && rm -rf /var/cache/apk/*

RUN mkdir -p /opt/app && \ 
    mkdir -p /opt/conf

COPY conf/. /opt/conf
COPY challenge/. /opt/app
    
WORKDIR /tmp
RUN wget https://www.haproxy.org/download/2.8/src/haproxy-2.8.1.tar.gz && \
    tar zxvf haproxy-*.tar.gz && cd haproxy-* && \
    make TARGET=linux-musl && \
    make install
RUN rm -rf *

WORKDIR /opt/conf

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install -I --no-cache-dir -r requirements.txt

WORKDIR /

EXPOSE 1337

CMD ["/usr/bin/supervisord", "-c", "/opt/conf/supervisord.conf"]