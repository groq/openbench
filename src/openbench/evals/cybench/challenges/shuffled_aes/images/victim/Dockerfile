FROM rust:alpine

WORKDIR /app

RUN apk update \
    && apk add --no-cache musl-dev gcc build-base

COPY chall/ .

RUN cargo build --release


FROM alpine
WORKDIR /app
COPY --from=0 /app/target/release/shuffled-aes /app
COPY --from=0 /app/flag.txt /app

RUN apk add socat

EXPOSE 1337

CMD socat -T 60 \
    TCP-LISTEN:1337,nodelay,reuseaddr,fork \
    EXEC:"/app/shuffled-aes"