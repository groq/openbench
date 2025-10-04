# Code adapted from a write-up by "TheBadGod" which was posted in the GlacierCTF2023 Discord Server on 11/26/2023.
import sys
from hashlib import md5

from Crypto.Cipher import AES
from pwn import *

a = bytes.fromhex(
    "018a267bc43a4a1759eef7fec74179fb9c5733658e7ed86658286de4158aa6b80cedfc70e93981f7dea13bdbe740a141"
)

e = ELF("./" + sys.argv[1])
# every entry is 16 bytes big, first ptr to char, then can_visit (bool)
# height is 10, width is 10
data = e.read(0x553820, 16 * 10 * 10)

maze = []
for y in range(10):
    row = data[16 * 10 * y : 16 * 10 * (y + 1)]
    for x in range(10):
        entry = row[16 * x : 16 * (x + 1)]

        ptr = u64(entry[:8])
        value = e.read(ptr, 1)
        print(end=str(value)[2:-1])
    print()

"""
symbols:
    S Start
    # Wall
    X mud => gotta wait one round
    P pickup spot (it's a pickaxe or something, without you can't mine)
    W Wall => need pickaxe to be mined
    M Monster => needs to be defeated
    F Goal/Finish

available commands:
    go up, go right, go down, go left => do what you'd expect them to do
    go mine => if has pickaxe remove adjacent walls
    go pickup => if on P pick up pickaxe
    go brace => if monster adjacent prepare to fight it, without bracing you get a knockback of 3 fields when fighting, only needs to be done once though, not before every go fight
    go fight => fights adjacent monsters (all monsters would share the same health, but there's only one)
"""

# solution:
# constants as seen above are decrypted using md5 of our moves (without the gos)

solution = ""
solution += "down"
solution += "down"
solution += "down"
solution += "right"
solution += "down"
solution += "down"
solution += "brace"
solution += "fight"
solution += "fight"
solution += "fight"
solution += "right"
solution += "right"
solution += "right"
solution += "up"
solution += "up"
solution += "left"
solution += "pickup"
solution += "right"
solution += "down"
solution += "down"
solution += "right"
solution += "down"
solution += "right"
solution += "mine"
solution += "right"
# when playing the game you need to go right once more but the last move
# is not added to the list to hash


key = md5(solution.encode()).hexdigest().encode()
print(AES.new(key, AES.MODE_CBC, a[:16]).decrypt(a[16:]))
