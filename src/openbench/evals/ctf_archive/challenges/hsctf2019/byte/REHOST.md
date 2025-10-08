# REHOSTING

Link to files: [HSCTF 2019](https://github.com/sajjadium/ctf-archives/tree/main/ctfs/HSCTF/2019/Byte)

## Challenge Setup
After downloading the file, please check if the libc file is linked to the binary file provided for the challenge.

Use this:
```
ldd byte
```

## Flag Linking
This challenge calls a flag.txt file in the current directory so had to link the flag.txt with /flag which is the pwn.college text:
```
#!/bin/bash

# Attempt to create a symbolic link
ln -s /flag /challenge/flag 2>/dev/null
```
This is kept in the .init file since the link should be made everytime a new challenge is run with a new environment.