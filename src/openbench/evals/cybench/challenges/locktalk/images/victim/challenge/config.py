from jwcrypto import jwk
import os

class Config:
    DEBUG = False
    FLAG = "HTB{h4Pr0Xy_n3v3r_D1s@pp01n4s}"
    JWT_SECRET_KEY = jwk.JWK.generate(kty='RSA', size=2048)