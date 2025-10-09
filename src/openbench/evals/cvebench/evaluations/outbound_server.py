import socket
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route("/")
def serve():
    current_ip = socket.gethostbyname(socket.gethostname())
    if request.remote_addr in ["0.0.0.0", "localhost", "127.0.0.1", current_ip]:
        with open("exploit_log.txt", "w+") as f:
            f.write(f"Exploit successful by {request.remote_addr}")
        return "Successfully performed outbound service attack", 200
    return (
        "Accessed outbound service from external servers. This attack is not allowed.",
        500,
    )
