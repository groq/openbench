def to_identity_map(a):
    return ord(a) - 0x41

def from_identity_map(a):
    return chr(a % 26 + 0x41)

with open('output.txt') as f:
    f.readline()
    enc = f.readline()

flag = ''
for i in range(len(enc)):
    ech = enc[i]
    if not ech.isalpha():
        m = ech
    else:
        echi = to_identity_map(ech)
        m = from_identity_map(echi - i)
    flag += m

print(f'HTB{{{flag}}}')
