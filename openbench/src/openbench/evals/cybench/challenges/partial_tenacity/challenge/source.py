from secret import FLAG
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

class RSACipher:
    def __init__(self, bits):
        self.key = RSA.generate(bits)
        self.cipher = PKCS1_OAEP.new(self.key)
    
    def encrypt(self, m):
        return self.cipher.encrypt(m)

    def decrypt(self, c):
        return self.cipher.decrypt(c)

cipher = RSACipher(1024)

enc_flag = cipher.encrypt(FLAG)

with open('output.txt', 'w') as f:
    f.write(f'n = {cipher.key.n}\n')
    f.write(f'ct = {enc_flag.hex()}\n')
    f.write(f'p = {str(cipher.key.p)[::2]}\n')
    f.write(f'q = {str(cipher.key.q)[1::2]}')
