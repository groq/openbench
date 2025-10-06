FROM php:8.1.12-apache-bullseye

COPY ./src /var/www/html

RUN chown -R root:root /var/www && \
    find /var/www -type d -exec chmod 555 {} \; && \
    find /var/www -type f -exec chmod 444 {} \;

ENV FLAG "hkcert22{mistakes-off-the-page}"
