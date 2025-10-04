
import mmh3
import pwn
from pwn import args


def forward_block(block):
    """
    initial transformation of 4 byte block in murmurhash3
    """
    n = int.from_bytes(block, 'little')
    n = (n * 0xcc9e2d51) & 0xffffffff
    n = (n >> 17) | (n << 15) & 0xffffffff
    n = (n * 0x1b873593) & 0xffffffff
    return n.to_bytes(4, 'little')


def invert_block(block):
    """
    invert a 4 byte block in murmurhash3
    """
    n = int.from_bytes(block, 'little')
    n = (n * 0x56ed309b) & 0xffffffff
    n = (n >> 15) | (n << 17) & 0xffffffff
    n = (n * 0xdee13bb1) & 0xffffffff
    return n.to_bytes(4, 'little')


def forward(blocks):
    """
    forward_block applied to all 4 byte chunks
    """
    return b''.join(forward_block(blocks[i:i + 4])
                    for i in range(0, len(blocks), 4))


def invert(blocks):
    """
    invert applied to all 4 byte chunks
    """
    return b''.join(invert_block(blocks[i:i + 4])
                    for i in range(0, len(blocks), 4))


DIFF = b'\x00\x00\x04\x00\x00\x00\x00\x80'


def collision8(text):
    return invert(bytes(i ^ j for i, j in zip(DIFF, forward(text))))


sample_valid_password = b'A' * 28 + b'Aa1!'
sample_colliding_password = collision8(
    sample_valid_password[:8]) + sample_valid_password[8:]

if args["HOST"]:
    HOST = args["HOST"]
if args["PORT"]:
    PORT = args["PORT"]

REM = pwn.remote(HOST, PORT)
REM.sendline('2') # store sample key
REM.sendline(sample_colliding_password.hex())
REM.sendline('3') # claim flag
REM.sendline(sample_valid_password.hex())
REM.sendline('4') # exit
print(REM.recvall().decode())
