import random
from collections import Counter
from math import log2

import pwn
from z3 import (
    And,
    BitVec,
    BitVecSort,
    Concat,
    Extract,
    Function,
    Implies,
    Not,
    Optimize,
    Product,
    RealSort,
    Sum,
    sat,
)

pwn.context.log_level = "critical"

if pwn.args["HOST"] and pwn.args["PORT"]:
	HOST = pwn.args["HOST"]
	PORT = pwn.args["PORT"]

REM = pwn.remote(HOST, PORT)
# REM = pwn.process("python3.11 chall.py", shell=True)

N = 6
B = 16

sbox = REM.recvline().strip().split()[-1]
sbox = sbox.decode("utf-8")
try:
    if len(sbox) % 2 != 0:
        raise ValueError("sbox string length is not even")

    sbox = [int(sbox[i : i + 2], 16) for i in range(0, len(sbox), 2)]
except ValueError as e:
    print(f"Error converting sbox to integers: {e}")
    exit(1)
REM.recvuntil(b"Get Flag\n")


def calculate_difference_table(sbox):
    # calculate the sbox difference distribution table
    n = len(sbox)
    bias = Counter()
    for inp_diff in range(n):
        for inp in range(n):
            out_diff = sbox[inp] ^ sbox[inp ^ inp_diff]
            bias[(inp_diff, out_diff)] += 1
    return bias


def print_bitrelations(inp_masks, out_masks):
    def bin_sep(val):
        v = bin(val)[2:].zfill(N * B)
        return "|".join(v[i : i + N] for i in range(0, N * B, N))

    rounds = len(out_masks)
    for i in range(rounds):
        imask, omask = inp_masks[i], out_masks[i]


def get_optimal_masks(sbox, pbox, num_rounds, bias, prune_level=0):
    n = int(log2(len(sbox)))
    num_blocks = len(pbox) // n
    # function which returns the bias for an input, output mask
    sboxf = Function("sbox", BitVecSort(n), BitVecSort(n), RealSort())
    s = Optimize()
    # inps[i][j] represent the input mask selected to jth sbox of ith layer
    inps = [
        [BitVec("r{}_i{}".format(r, i), n) for i in range(num_blocks)]
        for r in range(num_rounds + 1)
    ]
    # oups[i][j] represents the output mask selected from jth sbox of ith layer
    oups = [
        [BitVec("r{}_o{}".format(r, i), n) for i in range(num_blocks)]
        for r in range(num_rounds)
    ]

    def permutation(inp, oup, pbox):
        # bit wise permutation selected i.e constraints relating to pbox[inp]==oup
        pn = len(pbox)
        constraints = []
        for i, v in enumerate(pbox):
            constraints.append(
                Extract(pn - 1 - i, pn - 1 - i, inp)
                == Extract(pn - 1 - v, pn - 1 - v, oup)
            )
        return constraints

    for i in range(num_rounds):
        # The output of ith round is premutated to the input of i+1 th round
        s.add(permutation(Concat(oups[i]), Concat(inps[i + 1]), pbox))

    # It is not the case that all the input masks selected to the first layer are 0
    s.add(Not(And(*[inps[0][i] == 0 for i in range(num_blocks)])))
    for i in range(2**n):
        for j in range(2**n):
            # just some pruning of very small biases
            if bias[(i, j)] >= 2 ** (prune_level):
                s.add(sboxf(i, j) == bias[(i, j)])
            else:
                s.add(sboxf(i, j) == 0)
    for r in range(num_rounds):
        for i in range(num_blocks):
            # if sbox has input, it should have ouput
            s.add(Implies(inps[r][i] != 0, oups[r][i] != 0))
            # if sbox has no input it should not have any output
            s.add(Implies(inps[r][i] == 0, oups[r][i] == 0))
            # skip taking input/outputs with no bias
            s.add(
                Implies(
                    And(inps[r][i] != 0, oups[r][i] != 0),
                    sboxf(inps[r][i], oups[r][i]) != 0,
                )
            )
    s.check()  # initial check
    objectives = [
        # The reduced objective with the same maxima as the original objective
        # below but sum is easier to optimize
        Sum(
            [
                sboxf(
                    inps[i // num_blocks][i % num_blocks],
                    oups[i // num_blocks][i % num_blocks],
                )
                for i in range(num_blocks * num_rounds)
            ]
        ),
        # original objective i.e product of all bias terms selected
        # from each input/output mask
        # not that the bias is in range 0,2**n not the actual differential probability
        # from 0,1. So we need to divide each element by 2**n i.e 2**n times the number of
        # sboxes involved
        Product(
            [
                sboxf(
                    inps[i // num_blocks][i % num_blocks],
                    oups[i // num_blocks][i % num_blocks],
                )
                for i in range(num_blocks * num_rounds)
            ]
        )
        / ((2**n) ** (num_blocks * num_rounds)),
    ]
    s.maximize(objectives[0])
    results = []
    for i in range(num_blocks):
        s.pop()  # Removing any previously saved solver state
        s.push()  # Saving the current state of the solver.
        # the constraints added after it will be removed after calling solver.pop()
        for j in range(num_blocks):
            # Just the input to jth block should be non zero from the last layer
            if j == i:
                s.add(inps[-1][j] != 0)
            else:
                s.add(inps[-1][j] == 0)
        if s.check() == sat:
            m = s.model()
            inp_masks = [
                m.eval(Concat(inps[i])).as_long() for i in range(num_rounds + 1)
            ]
            oup_masks = [m.eval(Concat(oups[i])).as_long() for i in range(num_rounds)]
            total_bias = m.eval(objectives[1]).as_fraction()
            print_bitrelations(inp_masks, oup_masks)
            results.append((inp_masks, oup_masks, total_bias))
    return results


def get_encryptions(pts):
    pt_bytes = b"".join(i.to_bytes(12, "big") for i in pts)
    REM.sendline(b"1")
    REM.sendline(pt_bytes.hex().encode())
    data = REM.recvuntil(b"Get Flag\n")
    encs_bytes = bytes.fromhex(pwn.re.search(b"\n([0-9a-f]+)\n", data)[1].decode())
    encs_ints = [
        int.from_bytes(encs_bytes[i : i + 12], "big")
        for i in range(0, len(encs_bytes), 12)
    ]
    return encs_ints


def gen_pbox(s, n):
    return [(s * i + j) % (n * s) for j in range(s) for i in range(n)]


def rotate_left(val, shift, mod):
    shift = shift % mod
    return (val << shift | val >> (mod - shift)) & ((1 << mod) - 1)


def reverse_expand_key(key, rounds, inv_sbox):
    keys = [key]
    for _ in range(rounds):
        keys.append(rotate_left(sub(keys[-1], inv_sbox), -N - 1, B * N))
    return keys


def int_to_list(inp):
    return [(inp >> (i * N)) & ((1 << N) - 1) for i in range(B - 1, -1, -1)]


def list_to_int(lst):
    res = 0
    for i, v in enumerate(lst[::-1]):
        res |= v << (i * N)
    return res


def sub(inp: int, sbox):
    ct = 0
    for i in range(B):
        ct |= sbox[(inp >> (i * N)) & ((1 << N) - 1)] << (N * i)
    return ct


pbox = gen_pbox(N, B)
bias = calculate_difference_table(sbox)
inv_sbox = [sbox.index(i) for i in range(len(sbox))]

COUNT = 50000 // (B + 1)

pts = [random.randint(0, 2 ** (N * B) - 1) for _ in range(COUNT)]
encs0 = get_encryptions(pts)

recovered_key = [None] * B
characteristics = get_optimal_masks(sbox, pbox, 3, bias)
for pos, (inp_masks, out_masks, bias) in enumerate(characteristics):
    inp_mask, out_mask = inp_masks[0], inp_masks[-1]
    out_blocks = int_to_list(out_mask)
    diff_counts = Counter()
    encsi = get_encryptions([i ^ inp_mask for i in pts])
    for i in range(len(sbox)):
        key = [0] * B
        key[pos] = i
        key = list_to_int(key)
        for c1, c2 in zip(encs0, encsi):
            diff = sub(c1 ^ key, inv_sbox) ^ sub(c2 ^ key, inv_sbox)
            diff = int_to_list(diff)
            diff_counts[i] += out_blocks[pos] == diff[pos]
    recovered_key[pos] = diff_counts.most_common(1)[0][0]

original_key = reverse_expand_key(list_to_int(recovered_key), 5, inv_sbox)[-1]

REM.sendline(b"2")
REM.sendline(str(original_key).encode())
print(REM.recvall().decode()[11:])
