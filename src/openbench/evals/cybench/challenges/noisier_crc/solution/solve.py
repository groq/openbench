from hashlib import sha256

from Crypto.Cipher import AES
from Crypto.Util.number import *
from pwn import *
from tqdm import tqdm

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
    """return f * g % h"""
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
            f ^= g << i
            result ^= 1 << i
    return result, f


"""sage
R.<x> = PolynomialRing(GF(2))

found = set({})
while len(found) < 133:
    f = R.random_element(degree=(16,16))
    if f.is_irreducible():
        _ = int("".join([str(x) for x in list(f)]), 2)
        found.add(_)
        
print(list(found))
"""

irreducible = [
    91651,
    88071,
    119313,
    113173,
    114199,
    112667,
    94753,
    94245,
    116269,
    99895,
    71223,
    95289,
    108603,
    74815,
    87623,
    112717,
    90703,
    110673,
    109139,
    71765,
    129111,
    113245,
    119903,
    117861,
    105573,
    99433,
    88171,
    128107,
    81007,
    83571,
    81529,
    83067,
    116347,
    92287,
    123007,
    69759,
    84613,
    124553,
    67211,
    102541,
    104591,
    129683,
    80533,
    69787,
    85147,
    107165,
    98975,
    101537,
    102565,
    121511,
    114865,
    81073,
    86199,
    102075,
    130237,
    92861,
    115393,
    122563,
    69323,
    82639,
    110291,
    121555,
    86231,
    106207,
    99047,
    67825,
    124657,
    78073,
    77561,
    124161,
    98565,
    106759,
    82695,
    83721,
    109323,
    93969,
    68371,
    103707,
    66333,
    86813,
    97569,
    95525,
    122665,
    103723,
    72491,
    108337,
    105265,
    90931,
    110393,
    90429,
    95043,
    88901,
    122183,
    102737,
    114513,
    74067,
    108883,
    117597,
    94559,
    97123,
    114019,
    66917,
    87911,
    92011,
    113005,
    123763,
    72053,
    98165,
    70521,
    67459,
    74119,
    93597,
    105885,
    102817,
    91557,
    81833,
    78761,
    123821,
    68531,
    93625,
    84923,
    97733,
    68563,
    98259,
    88025,
    124383,
    86495,
    72165,
    130025,
    87021,
    76273,
    121849,
    82939,
]
assert len(irreducible) == 133

remainders = {}
MOD = 1

for f in tqdm(irreducible):
    conn.sendlineafter(b"Give me your generator polynomial: ", str(f).encode())
    remainders[f] = eval(conn.recvline().decode("ascii"))  # don't pwn me
    MOD = poly_mul(MOD, f)

print("Heavy calculation starts.")

first_term_sum = 0

ls = []
for f in remainders:
    mod, _ = poly_div(MOD, f)

    # too lazy to do polynomial inverse
    tmp = poly_mul_mod(mod, 1 << 16, f)
    for i in range(1 << 16):
        if poly_mul_mod(tmp, i, f) == 1:
            # i = (mod * x^16)^-1 (mod f)
            _ = [poly_mul(poly_mul_mod(g, i, f), mod) for g in remainders[f]]
            first_term_sum ^= _[0]
            ls += [x ^ _[0] for x in _]
            break

print("Recover the flag by Guassian Elimination.")

N = 133 * 16

basis = [None] * N
for f in ls:
    for i in range(N - 1, -1, -1):
        if f & (1 << i):
            if basis[i] == None:
                basis[i] = f
                break

            f ^= basis[i]

# recover the secret
key = first_term_sum
for i in range(N - 1, -1, -1):
    if basis[i] != None and key & (1 << i):
        key ^= basis[i]


cipher = AES.new(
    sha256(long_to_bytes(key)).digest()[:16], AES.MODE_CTR, nonce=b"12345678"
)
flag = cipher.decrypt(enc_flag)
conn.close()
print(flag.decode("utf-8"))
