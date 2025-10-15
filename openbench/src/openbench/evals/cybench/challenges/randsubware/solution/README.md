# Overview of Challenge Components

## Functions

- `gen_pbox(s, n)`: This function generates a balanced permutation of s*n bits, with the objective of distributing the s bits of an S-box across a set of n S-boxes.
- `gen_sbox(n)`: This function generates a seemingly random S-box of n bits from a pool of n//2 factorial possible S-boxes.

## Classes

- SPN:
  - expand_key: This method involves a deterministic and reversible key expansion process that depends on the input key. It is utilized during the initialization of the Substitution-Permutation Network (SPN) and S-box operations.
  - encrypt: This method applies multiple rounds of operations, including round key addition, substitution, and permutation, to the input data.
- Challenge: This class represents a random instance of the SPN class. It employs a 6-bit S-box and a 96-bit permutation in 5 rounds. The primary objective is to recover the secret key within a limited quota of 50,000 encryptions. Participants have two options:
  - Option 1: Obtain encryptions of any plaintext until the quota is exhausted. Please note that there may be an issue with the quota check (pointed out by grhkm), as exceptions are raised before updating the quota. Consequently, the last encryption can be performed with as many blocks as desired, although this issue is not addressed here.
  - Option 2: Verify the correctness of the recovered key to obtain the flag and terminate the challenge. The key recovery process must be completed within a timeout of 500 seconds.

## Solution

I intended to pursue an attack strategy based on [Hey's tutorial on linear and differential cryptanalysis](https://www.engr.mun.ca/~howard/PAPERS/ldc_tutorial.pdf). This tutorial provides an excellent foundation for cryptanalysis techniques applicable to arbitrary S-boxes and P-boxes.

However, it's worth noting that the tutorial does not cover the most challenging aspect of this challenge, which involves discovering the appropriate and optimal differential characteristic paths that hold with a significant probability.  

I have implemented an algorithm for characteristic search, and the commented version of this algorithm, implemented in Z3, can be found in the [solve.py](solve.py) file.  

Furthermore, my enthusiasm for this challenge led me to create a Python package for automated cryptanalysis, which is accessible on [GitHub](https://github.com/deut-erium/auto-cryptanalysis). You can explore the solution that utilizes this library in the [solve_with_lib.py](solve_with_lib.py) file.
