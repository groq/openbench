import time 
from pwn import *


def getParity(ct: int): # odd: < B, even: >= B
	p.sendline(b"1")
	p.recvuntil(b"Ciphertext in hex: ")
	p.sendline(hex(ct)[2:])
	x, y = eval(p.recvline())
	return (x + y) % 2

def getInfo(f: int):
	c = enc_sol * pow(f, e, n) % n
	if getParity(c):
		return "<"
	else:
		return ">"

found_workable_n = False
while not found_workable_n:

	HOST = args["HOST"]
	PORT = args["PORT"]
	p = remote(HOST, PORT)

	welcome = p.recvline() # Welcome message

	n_raw = p.recvline().split(b"=")[-1]
	n_raw_split = n_raw.split(b"=")[-1]
	n = int(n_raw_split)
	e = int(p.recvline().split(b"=")[-1])
	enc_sol = int(p.recvline()[2:], 16)

	# Requirement: n >= 2B
	if n.bit_length() == 1024:
		# Ok to proceed
		found_workable_n = True
		break
	else:
		# Close connection, try again.
		p.close() 
		continue

B = (1 << 1022)

# the following only works when sol_int > 2^127, which happens with probability 1/2

sol_min = 2 ** 127
sol_max = 2 ** 128

while sol_max - sol_min > 100:
	tmp = 3 * B // 2 // (sol_max - sol_min)
	i = (tmp * sol_min - B // 4) // n
	boundary = (sol_min + sol_max) // 2
	f = (i * n + B) // boundary
	assert i * n + B // 4 <= sol_min * f <= sol_max * f <= i * n + B * 2
	assert sol_min <= boundary <= sol_max

	if getInfo(f) == "<":
		sol_max = boundary
	else:
		sol_min = boundary + 1

while sol_min <= sol_max and pow(sol_min, e, n) != enc_sol:
	sol_min += 1

assert sol_min <= sol_max, "Solution is out of range"

def toPath(x: int):
	directions = "LRUD"
	s = bin(x)[2:]
	if len(s) % 2 == 1:
		s = "0" + s

	path = ""
	for i in range(0, len(s), 2):
		path += directions[int(s[i:i+2], 2)]
	return path

sol = toPath(sol_min) # Don't need to pad L bc we assume sol > 2^127

p.sendline("2")
p.sendline(sol)

time.sleep(0.2)
# Receive and discard prompt
p.recvline().decode()
time.sleep(0.2)

flag = p.recvline().decode()
print(flag)

# Close the connection
p.close()
