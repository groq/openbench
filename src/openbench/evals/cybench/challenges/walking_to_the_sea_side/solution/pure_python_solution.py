import base64
import itertools
import json
import os
import re

from Crypto.Cipher import AES
from Crypto.Hash import SHA512
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.number import long_to_bytes
from pwn import log, process, remote

MAX_STEPS = 5
BASE_CURVE = 0
FLAG_REGEX = re.compile(r"gctf{.*}")

custom_primes = []
while (2 * MAX_STEPS + 1) ** len(custom_primes) < 0xFF << (62 * 4):
    custom_primes.append(3)
custom_primes.append(5)
custom_primes.append(7)

supersingular_curves = [
    # these are all the curves that are supersingular in F_p = 419
    0,
    158,
    410,
    368,
    404,
    75,
    144,
    191,
    174,
    413,
    379,
    124,
    199,
    390,
    29,
    220,
    295,
    40,
    6,
    245,
    228,
    275,
    344,
    15,
    51,
    9,
    261,
]


def main() -> int:

    # alice_conn = process(
    #     ["sage", "../dist/chal.sage"],
    #    env={
    #   "HOME": os.environ["HOME"],
    #  "USERNAME": "alice"}
    # )

    HOSTNAME = os.getenv("HOST", "alice")
    PORT = os.getenv("PORT", 1337)
    # alice_conn = remote(HOSTNAME, PORT)
    # alice_conn = process(['python3', 'chal.sage.py'])
    alice_conn = remote(HOSTNAME, PORT)

    log.info("Tell her that we are listening")
    alice_conn.recvuntil(b"Can you listen?")
    alice_conn.recvuntil(b"> ")
    alice_conn.sendline(b"yes")

    log.info("Tell her that we want to use custom primes")
    alice_conn.recvuntil(b"Are you OK with using")
    alice_conn.recvuntil(b"> ")
    alice_conn.sendline(b"no")

    log.info("Sending her our prepared comma seperated list")
    alice_conn.recvuntil(b"Please send me a comma separated list consisting of primes")
    alice_conn.recvuntil(b"> ")
    alice_conn.sendline(",".join(str(prime) for prime in custom_primes).encode())

    log.info("Waiting for her secret and sending a dummyvalue as bobs")
    alice_conn.recvuntil(b"Can you send me yours?")
    alice_conn.recvuntil(b"> ")
    alice_conn.sendline(f"{supersingular_curves[-1]}".encode())

    log.info("Receiving her encrypted output")
    alice_conn.recvuntil(b"Please don't share it with Eve")
    alice_conn.recvline()
    result = json.loads(alice_conn.recvline().decode())
    log.info(result)
    alice_conn.close()

    log.info("Decode her output using the precomputet supersingular curves")
    for supersingular_curve in supersingular_curves:
        log.info(f"Trying curve {supersingular_curve}")
        try:
            aes_secret_key = PBKDF2(
                long_to_bytes(supersingular_curve),
                b"secret_salt",
                count=1000000,
                hmac_hash_module=SHA512,
            )

            cipher = AES.new(
                aes_secret_key, AES.MODE_GCM, nonce=base64.b64decode(result["Nonce"])
            )
            plaintext = cipher.decrypt_and_verify(
                base64.b64decode(result["CT"]),
                base64.b64decode(result["Tag"]),
            )
            log.success(f"Decoded Message: {plaintext.decode()}")
            flag = FLAG_REGEX.findall(plaintext.decode())[0]
            flag = flag[flag.find("gctf") :]
            print(flag)
            return 0

        except ValueError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
