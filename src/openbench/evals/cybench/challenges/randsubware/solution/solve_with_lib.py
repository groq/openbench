from cryptanalysis import DifferentialCryptanalysis
import pwn


pwn.context.log_level = 0

# REM = pwn.process("python3.11 chall.py",shell=True)
REM = pwn.remote("chals.sekai.team", 3037)

SBOX = REM.recvline().strip().split()[-1]
SBOX = [int(SBOX[i:i + 2], 16) for i in range(0, len(SBOX), 2)]
REM.recvuntil(b'Get Flag\n')


def get_encryptions(pts):
    pt_bytes = b"".join(i.to_bytes(12, 'big') for i in pts)
    REM.sendline(b"1")
    REM.sendline(pt_bytes.hex().encode())
    data = REM.recvuntil(b'Get Flag\n')
    encs_bytes = bytes.fromhex(pwn.re.search(b'\n([0-9a-f]+)\n', data)[1].decode())
    encs_ints = [int.from_bytes(encs_bytes[i:i + 12], 'big')
                 for i in range(0, len(encs_bytes), 12)]
    return encs_ints


def gen_pbox(s, n):
    return [(s * i + j) % (n * s) for j in range(s) for i in range(n)]


def rotate_left(val, shift, mod):
    shift = shift % mod
    return (val << shift | val >> (mod - shift)) & ((1 << mod) - 1)

def reverse_expand_key(spn, key, rounds):
    keys = [key]
    for _ in range(rounds):
        keys.append(rotate_left(
            spn.inv_sub(keys[-1]), -spn.box_size - 1, spn.block_size))
    return keys

PBOX = gen_pbox(6, 16)
cryptanal = DifferentialCryptanalysis(SBOX, PBOX, 5)
cryptanal.batch_encrypt = get_encryptions

diff_characteristics = cryptanal.characteristic_searcher.search_exclusive_masks(
    prune_level=1)

last_round_key_blocks = cryptanal.find_last_roundkey(
    diff_characteristics, 50000 // 16)
last_round_key = cryptanal.list_to_int(last_round_key_blocks)
original_key = reverse_expand_key(cryptanal, last_round_key, 5)[-1]

REM.sendline(b"2")
REM.sendline(str(original_key).encode())
print(REM.recvall())
