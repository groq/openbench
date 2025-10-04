# Description

In this challenge a modified AES is used in CTR mode.
You receive an encrypted flag and are able to query the encryption of any plaintext.

## Solution

The cipher performs 10 rounds of (SubBytes, AddRoundKey) and only then 10 rounds of (ShiftRows, MixColumns, AddRoundKey).
The final 10 rounds only consist of linear operations.
Therefore, we can reverse them by applying (InverseMixColumns, InverseShiftRows) 10 times.
This leaves us with a cipher that operates on each byte individually.
Now, we can build a lookup table for each input/output byte at each of the 16 offsets.
We query ~1000 plaintexts until the tables are populated enough to decrypt the flag.

The exploit script can be run using

```bash
python3 ./exploit.py
```
