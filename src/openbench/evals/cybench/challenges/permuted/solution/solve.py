from Crypto.Cipher import AES
from Crypto.Util.number import long_to_bytes

from hashlib import sha256

from sympy.ntheory.modular import crt

# DLP found in https://www.researchgate.net/publication/326514386_Cryptanalysis_of_a_Proposal_Based_on_the_Discrete_Logarithm_Problem_Inside_Sn
class Permutation:
    def __init__(self, mapping):
        self.length = len(mapping)

        assert set(mapping) == set(range(self.length))     # ensure it contains all numbers from 0 to length-1, with no repetitions
        self.mapping = list(mapping)

    def __call__(self, *args, **kwargs):
        idx, *_ = args
        assert idx in range(self.length)
        return self.mapping[idx]

    def __mul__(self, other):
        ans = []

        for i in range(self.length):
            ans.append(self(other(i)))

        return Permutation(ans)

    def __pow__(self, power, modulo=None):
        ans = Permutation.identity(self.length)
        ctr = self

        while power > 0:
            if power % 2 == 1:
                ans *= ctr
            ctr *= ctr
            power //= 2

        return ans

    def __str__(self):
        return str(self.mapping)

    def cycles(self):
        # this was added!
        cycles = []
        used = set()

        for i in self.mapping:
            if i in used:
                continue

            curr_cycle = [i]
            used.add(i)

            idx = self(i)
            while idx not in used:
                curr_cycle.append(idx)
                used.add(idx)
                idx = self(idx)

            cycles.append(curr_cycle)

        return cycles

    def identity(length):
        return Permutation(range(length))



def dlp(g, h):
    # g is base
    # h is result
    g_cycles = g.cycles()
    h_cycles = h.cycles()

    # print('g cycles:', g_cycles)
    # print('h cycles:', h_cycles)

    G = []
    H = []

    for i in range(g.length):
        for j, c in enumerate(g_cycles):
            if i in c:
                G.append((j, c.index(i)))

        for j, c in enumerate(h_cycles):
            if i in c:
                H.append((j, c.index(i)))

    # print('G:', G)
    # print('H:', H)

    First = []
    Second = []

    for c in h_cycles:
        First.append(c[0])
        Second.append(c[1 % len(c)])

    # print('first:', First)
    # print('second:', Second)

    D = []
    L = []
    for i in range(len(Second)):
        dist = G[Second[i]][1] - G[First[i]][1]
        D.append(dist)
        L.append(len(h_cycles[i]))

    # print('D:', D)
    # print('L:', L)

    alpha = crt(L, D)

    return int(alpha[0])


# solve!
with open('output.txt') as f:
    exec(f.read())

g = Permutation(g)
A = Permutation(A)
B = Permutation(B)

a = dlp(g, A)

# decrypt
C = B**a

sec = tuple(C.mapping)
sec = hash(sec)
sec = long_to_bytes(sec)

hash = sha256()
hash.update(sec)

key = hash.digest()[16:32]
iv = b"mg'g\xce\x08\xdbYN2\x89\xad\xedlY\xb9"

cipher = AES.new(key, AES.MODE_CBC, iv)

decrypted = cipher.decrypt(c)

# Strip padding
padding_length = decrypted[-1]  # get the last byte, which indicates the padding length
decrypted = decrypted[:-padding_length]  # remove the padding

# Convert bytes to string
decrypted_string = decrypted.decode('utf-8')
print()

print(decrypted_string)

# ~70 seconds
