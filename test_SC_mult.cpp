#include <stdio.h>
#include "./llmc/SC.h"

int main() {
    // Example input values (outside [0,1), assuming a known range, e.g., [-10,10])
    float a = -0.125765;
    float b = 0.012414;
    float max_abs_val = 1.0;

    float result;

    // Define the expected input range for normalization.
    // For example, if we expect x ∈ [-10, 10]:
    // float a = 0.837400;
    // float b = 2.795600;
    // float max_abs_val = 2.795600;

    // Call the top-level function for stochastic multiplication.
    // It normalizes the inputs, converts them to fixed point, generates stochastic bitstreams,
    // performs bitwise AND (multiplication), and converts the result back to a float.
    result = SC_mult(a, b, max_abs_val);

    // Print the result. Note: Due to the stochastic nature, the result is an approximation.
    printf("binary multiplication result for %f * %f = %f\n", a, b, a * b);
    printf("Stochastic multiplication result for %f * %f = %f\n", a, b, result);
    printf("MAE: %f\n", abs(a * b - result));

    return 0;
}