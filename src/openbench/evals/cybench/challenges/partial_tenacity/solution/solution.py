from math import sqrt
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def load_data():
    with open('output.txt') as f:
        n = int(f.readline().split(' = ')[1])
        ct = bytes.fromhex(f.readline().split(' = ')[1])
        hint_p = int(f.readline().split(' = ')[1])
        hint_q = int(f.readline().split(' = ')[1])
    return n, ct, hint_p, hint_q

def decrypt(p, q, n, ct):
    e = 0x10001
    d = pow(e, -1, (p-1)*(q-1))
    key = RSA.construct((n, e, d))
    flag = PKCS1_OAEP.new(key).decrypt(ct)
    return flag

def create_masks(primelen):
    pmask = ''.join(['1' if i % 2 == 0 else '0' for i in range(primelen)])
    qmask = ''.join(['1' if i % 2 == 1 else '0' for i in range(primelen)])
    return pmask, qmask

def bruteforce_digit(i, n, known_prime, prime_to_check, hint_prime):
    msk = 10**(i+1)
    known_prime = 10**i * (hint_prime % 10) + known_prime
    for d in range(10):
        test_prime = 10**i * d + prime_to_check
        if n % msk == known_prime * test_prime % msk:
            updated_prime_to_check = test_prime			    # correct candidate! update the unknown prime
            updated_hint_prime = hint_prime // 10			# move on to the next digit
            return known_prime, updated_prime_to_check, updated_hint_prime

def factor(n, p, q, hp, hq, pmask, qmask, prime_len):
    for i in range(prime_len):
        if pmask[-(i+1)] == '1':
            p, q, hp = bruteforce_digit(i, n, p, q, hp)
        else:
            q, p, hq = bruteforce_digit(i, n, q, p, hq)

    assert n == p * q

    return p, q

def pwn():
    n, ct, hint_p, hint_q = load_data()
    prime_len = len(str(int(sqrt(n))))
    pmask, qmask = create_masks(prime_len)
    p, q = factor(n, 0, 0, hint_p, hint_q, pmask, qmask, prime_len)
    flag = decrypt(p, q, n, ct)
    print(flag.decode('utf-8'))

if __name__ == '__main__':
    pwn()