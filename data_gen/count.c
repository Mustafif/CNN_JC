#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

bool hasTenZeros(const char *str, int n) {
    for (int i = 0; i <= n - 10; i++) {
        bool allZeros = true;
        for (int j = 0; j < 10; j++) {
            if (str[i + j] != '0') {
                allZeros = false;
                break;
            }
        }
        if (allZeros) {
            return true;
        }
    }
    return false;
}

double estimateProbability(int n, int trials) {
    int count = 0;
    for (int t = 0; t < trials; t++) {
        char str[n + 1];
        for (int i = 0; i < n; i++) {
            str[i] = (rand() % 2) ? '1' : '0';
        }
        str[n] = '\0';
        if (hasTenZeros(str, n)) {
            count++;
        }
    }
    return (double)count / trials;
}

int main() {
    srand(time(NULL)); // Seed the random number generator
    int n = 1000;
    int trials = 1000000; // Number of trials for Monte Carlo simulation

    double probability = estimateProbability(n, trials);
    printf("Estimated probability p(%d) = %f\n", n, probability);

    return 0;
}
