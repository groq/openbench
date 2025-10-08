# REHOSTING

Link to files: [HSCTF 2019](https://github.com/sajjadium/ctf-archives/tree/main/ctfs/HSCTF/2019/Aria_Writer)

## Challenge Setup
After downloading the file, please check if the libc file is linked to the binary file provided for the challenge.

Use this:
```
ldd aria-writer
```
Since the file only contains 1 libc file, it should work with the current environment.

## Troubleshooting
Use patchelf to make sure the current challenge directory is used for the libc files:
```
patchelf --set-rpath /challenge /challenge/aria-writer
```