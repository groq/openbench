#!/bin/bash

# Write the C program
cat << 'EOF' > caesars-revenge.c
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (setresuid(0, 0, 0) < 0) {
        perror("setresuid");
        exit(EXIT_FAILURE);
    }
    char *challenge_path = "/challenge/caesars";
    execv(challenge_path, argv);
    perror("execv");
    return EXIT_FAILURE;
}
EOF

# Compile the C program
gcc -o caesars-revenge caesars-revenge.c

# Set ownership and permissions
sudo chown root:root caesars-revenge
sudo chmod 4755 caesars-revenge

# Clean up the source file
rm caesars-revenge.c
