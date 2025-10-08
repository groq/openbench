# REHOSTING

Link to files: [HSCTF 2019](https://github.com/hsncsclub/HSCTF-6-Problems/tree/master/misc/broken-repl)

## Challenge Setup
There is only one python file with no dependenices, so it should run with this command-
```
python repl.py
```

## Flag Linking
This challenge calls a flag.txt file in the current directory so had to link the flag.txt with /flag which is the pwn.college text:
```
#!/bin/bash

# Attempt to create a symbolic link
ln -s /flag /challenge/flag 2>/dev/null
```
This is kept in the .init file since the link should be made everytime a new challenge is run with a new environment.