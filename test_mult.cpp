#include <stdio.h>
#include "mult.h"

int main() {
    // Example input values (outside [0,1), assuming a known range, e.g., [-10,10])
    float a = 2.5f;
    float b = 1.5f;
    float result;

    // Call the top-level function for stochastic multiplication.
    // It normalizes the inputs, converts them to fixed point, generates stochastic bitstreams,
    // performs bitwise AND (multiplication), and converts the result back to a float.
    mult(a, b, &result);

    // Print the result. Note: Due to the stochastic nature, the result is an approximation.
    printf("Conventianl multiplication result for %f * %f = %f\n", a, b, result);

    return 0;
}