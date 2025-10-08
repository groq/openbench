#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // Set the real, effective, and saved user IDs to root (0)
    if (setresuid(0, 0, 0) < 0) {
        perror("setresuid");
        exit(EXIT_FAILURE);
    }

    // Path to the challenge binary
    char *challenge_path = "/challenge/caesars";

    // Execute the challenge
    execv(challenge_path, argv);

    // If execv returns, there was an error
    perror("execv");
    return EXIT_FAILURE;
}

