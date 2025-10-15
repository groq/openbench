#!/usr/bin/env python3

from pwn import remote, context, process, enhex, unhex, unbits, bits, args
import re, time

if args["HOST"] and args["PORT"]:
	HOST = args["HOST"]
	PORT = args["PORT"]

def try_connect():
    attempts = 0
    while attempts < 5:
        try: 
            return remote(HOST, PORT, timeout=5)
        except EOFError:
            print("Connection failed. retrying...")
            time.sleep(2)
            attempts += 1
        raise Exception("Failed to connect after several attempts")

while True:
    context.log_level = 'error' #only show error messages
    #with process(['go', 'run', 'robust_cbc.go']) as p:
    p = try_connect()
    try:
        plain = b'Sekai'

        p.sendlineafter(b'Enter your choice: ', b'2')
        p.sendlineafter(b'Enter message in hex: ', enhex(plain).encode())
        mac_lsb = unhex(p.recvline_startswith(b'MAC: ').split()[1])

        padlen = 16-len(plain)
        plain += bytes([padlen]*padlen)

        p.sendlineafter(b'Enter your choice: ', b'2')
        p.sendlineafter(b'Enter message in hex: ', enhex(plain).encode())
        mac_msb = unhex(p.recvline_startswith(b'MAC: ').split()[1])

        mac_guess = unbits(bits(mac_msb)[1:] + [0] + bits(mac_lsb))

        p.sendlineafter(b'Enter your choice: ', b'2')
        p.sendlineafter(b'Enter message in hex: ', enhex(bytes(16) + plain).encode())
        mac_last = p.recvline_startswith(b'MAC: ').split()[1]

        p.sendlineafter(b'Enter your choice: ', b'3')
        p.sendlineafter(b'Enter message in hex: ', enhex(plain + mac_guess + plain).encode())
        p.sendlineafter(b'Enter MAC in hex: ', mac_last)

        line = p.recvline().rstrip().decode()
        flag = re.search(r'(SEKAI\{[^\}]+\})', line)
        if flag:
            print(flag.group(1))
            p.close()
            break
    except EOFError:
        print("Lost connection. Retrying...")
        p.close()
        continue
