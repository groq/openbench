import hashlib
import base64
import os
import binascii
from io import StringIO
from array import array

aes_sbox = array('B', bytes.fromhex(
    '637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0'
    'b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275'
    '09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf'
    'd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2'
    'cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb'
    'e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08'
    'ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e'
    'e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16'
))

def galois_multiply(a, b):
    """Galois Field multiplicaiton for AES"""
    p = 0
    while b:
        if b & 1:
            p ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x1b
        b >>= 1

    return p & 0xff

# Precompute the multiplication tables for encryption
gf_mul_by_2  = array('B', [galois_multiply(x,  2) for x in range(256)])
gf_mul_by_3  = array('B', [galois_multiply(x,  3) for x in range(256)])
# ... for decryption
gf_mul_by_9  = array('B', [galois_multiply(x,  9) for x in range(256)])
gf_mul_by_11 = array('B', [galois_multiply(x, 11) for x in range(256)])
gf_mul_by_13 = array('B', [galois_multiply(x, 13) for x in range(256)])
gf_mul_by_14 = array('B', [galois_multiply(x, 14) for x in range(256)])


def mix_columns(block):
    """MixColumns step. Mixes the values in each column"""

    mul_by_2 = gf_mul_by_2
    mul_by_3 = gf_mul_by_3

    for i in range(4):
        col = i * 4

        v0, v1, v2, v3 = (block[col], block[col + 1], block[col + 2],
                          block[col + 3])

        block[col  ] = mul_by_2[v0] ^ v3 ^ v2 ^ mul_by_3[v1]
        block[col+1] = mul_by_2[v1] ^ v0 ^ v3 ^ mul_by_3[v2]
        block[col+2] = mul_by_2[v2] ^ v1 ^ v0 ^ mul_by_3[v3]
        block[col+3] = mul_by_2[v3] ^ v2 ^ v1 ^ mul_by_3[v0]


def mix_columns_inv(block):
    """Similar to mix_columns above, but performed in inverse for decryption."""

    mul_9  = gf_mul_by_9
    mul_11 = gf_mul_by_11
    mul_13 = gf_mul_by_13
    mul_14 = gf_mul_by_14

    for i in range(4):
        col = i * 4

        v0, v1, v2, v3 = (block[col], block[col + 1], block[col + 2],
                          block[col + 3])

        block[col  ] = mul_14[v0] ^ mul_9[v3] ^ mul_13[v2] ^ mul_11[v1]
        block[col+1] = mul_14[v1] ^ mul_9[v0] ^ mul_13[v3] ^ mul_11[v2]
        block[col+2] = mul_14[v2] ^ mul_9[v1] ^ mul_13[v0] ^ mul_11[v3]
        block[col+3] = mul_14[v3] ^ mul_9[v2] ^ mul_13[v1] ^ mul_11[v0]

def shift_rows(b):
    b[1], b[5], b[ 9], b[13] = b[ 5], b[ 9], b[13], b[ 1]
    b[2], b[6], b[10], b[14] = b[10], b[14], b[ 2], b[ 6]
    b[3], b[7], b[11], b[15] = b[15], b[ 3], b[ 7], b[11]

def shift_rows_inv(b):
    b[ 5], b[ 9], b[13], b[ 1] = b[1], b[5], b[ 9], b[13]
    b[10], b[14], b[ 2], b[ 6] = b[2], b[6], b[10], b[14]
    b[15], b[ 3], b[ 7], b[11] = b[3], b[7], b[11], b[15]
