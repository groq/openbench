from pwn import *
from Crypto.Util.number import *
from Crypto.Cipher import AES
from hashlib import sha256


if args["HOST"] and args["PORT"]:
	host = args["HOST"]
	port = args["PORT"]
conn = remote(host, port)

conn.recvuntil(b"Encrypted flag: ")
enc_flag = bytes.fromhex(conn.recvline().decode("ascii").strip())

def poly_mul(f, g):
	result = 0
	while g:
		if g & 1:
			result ^= f
		f <<= 1
		g >>= 1
	return result

def poly_mul_mod(f, g, h):
	""" return f * g % h"""
	_, f = poly_div(f, h)
	_, g = poly_div(g, h)
	
	result = 0
	while g:
		if g & 1:
			result ^= f
		f <<= 1
		if f ^ h < f:
			f ^= h
		g >>= 1
	return result

def poly_div(f, g):
	result = 0
	for i in range(f.bit_length() - g.bit_length() + 1, -1, -1):
		if f ^ (g << i) < f:
			f ^= (g << i)
			result ^= (1 << i)
	return result, f

"""sage
R.<x> = PolynomialRing(GF(2))

found = set({})
while len(found) < 64:
    f = R.random_element(degree=(16,16))
    if f.is_irreducible():
        _ = int("".join([str(x) for x in list(f)]), 2)
        found.add(_)
        
print(list(found))
"""

irreducible = [87939, 97539, 127879, 122251, 103819, 67597, 84367, 103057, 95249, 129683, 122261, 125335, 94487, 129943, 77335, 116251, 112283, 91171, 114341, 118439, 90919, 78761, 110633, 71723, 65579, 91951, 85681, 66867, 114999, 90683, 80317, 104769, 100933, 71495, 94023, 90697, 87371, 72139, 85457, 119763, 96475, 70107, 106075, 89055, 130911, 87009, 125537, 111587, 127459, 119013, 100967, 87911, 124009, 86381, 66157, 119407, 72307, 128885, 120053, 125687, 103029, 73589, 68221, 93823]

remainders = {}
MOD = 1

for f in irreducible:
	conn.sendlineafter(b"Give me your generator polynomial: ", str(f).encode())
	remainders[f] = eval(conn.recvline().decode("ascii"))  # don't pwn me
	MOD = poly_mul(MOD, f)

ls = []
for f in remainders:
	mod, _ = poly_div(MOD, f)

	# too lazy to do polynomial inverse
	tmp = poly_mul_mod(mod, 1 << 16, f)
	for i in range(1 << 16):
		if poly_mul_mod(tmp, i, f) == 1:
			# i = (mod * x^16)^-1 (mod f)
			ls.append([poly_mul(poly_mul_mod(g, i, f), mod) for g in remainders[f]])
			break

# `ls` will be a list of 3-tuple. We will pick an element per tuple, and the sum will be the secret.
# We have a total of 3 * 64 polynomials of degree 16 * 64, and the secret (degree = 512) is in the space they span.

basis = [None] * 1024
for _ in ls:
	for f in _:
		for i in range(1023, -1, -1):
			if f & (1 << i):
				if basis[i] == None:
					basis[i] = f
					break
				
				f ^= basis[i]

# recover the secret
for i in range(512):
	if basis[i] != None:
		key = basis[i]
		cipher = AES.new(sha256(long_to_bytes(key)).digest()[:16], AES.MODE_CTR, nonce=b"12345678")
		flag = cipher.decrypt(enc_flag)
		print(flag)
