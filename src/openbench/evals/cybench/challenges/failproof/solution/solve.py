import hashlib
from os import urandom as rand
from ortools.sat.python import cp_model
from pwn import remote,args

if args["HOST"] and args["PORT"]:
    HOST = args["HOST"]
    PORT = args["PORT"]
    print(f'Running solver remotely at {HOST} {PORT}\n')

def int_to_bits(i, size):
    return list(map(int, bin(i)[2:].zfill(size)))


def bits_to_bytes(bits):
    intv = int("".join(map(str, bits)), 2)
    return int.to_bytes(intv, (intv.bit_length() + 7) // 8, "big")


def gen_pubkey(secret: bytes, hasher=hashlib.sha512) -> list:
    def hash(m):
        return hasher(m).digest()

    state = hash(secret)
    pubkey = []
    for _ in range(len(hash(b"0")) * 4):
        pubkey.append(int.from_bytes(state, "big"))
        state = hash(state)
    return pubkey


def get_solution(A, B, max_time=5):
    model = cp_model.CpModel()
    X = [model.NewIntVar(0, 1, "x_" + str(i)) for i in range(len(A[0]))]
    for row, val in zip(A, B):
        model.Add(sum(i * j for i, j in zip(X, row)) == val)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [solver.Value(i) for i in X]


def get():
    REM = remote(HOST, PORT)
    secret = bytes.fromhex(REM.recvline().strip().decode())
    B = REM.recvline()
    REM.close()
    A = [int_to_bits(i, 256) for i in gen_pubkey(secret, hashlib.sha256)]
    return A, eval(B)


A, B = get()
flag = [None] * len(B)
while not all(flag):
    for i, b in enumerate(B):
        if not flag[i]:
            if x := get_solution(A, b):
                flag_bytes = bits_to_bytes(x)
                flag[i] = flag_bytes
                print(flag)
    A, B = get()

print(b"".join(flag))
