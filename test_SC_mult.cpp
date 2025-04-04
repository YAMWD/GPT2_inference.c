#include <stdio.h>
#include "./llmc/SC.h"

int main() {
    // Example input values (outside [0,1), assuming a known range, e.g., [-10,10])
    float a = 0.8374;
    float b = 2.7956;
    float result;

    // Define the expected input range for normalization.
    // For example, if we expect x ∈ [-10, 10]:
    float max_abs_val = 2.7956f;

    // Set initial seed values for the LFSR used in stochastic bitstream generation.
    ap_uint<24> seed1 = 0xACE1;  // Example seed value for input 'a'
    ap_uint<24> seed2 = 0xBEEF;  // Example seed value for input 'b'

    // Call the top-level function for stochastic multiplication.
    // It normalizes the inputs, converts them to fixed point, generates stochastic bitstreams,
    // performs bitwise AND (multiplication), and converts the result back to a float.
    SC_mult(a, b, &result, max_abs_val, seed1, seed2);

    // Print the result. Note: Due to the stochastic nature, the result is an approximation.
    printf("binary multiplication result for %f * %f = %f\n", a, b, a * b);
    printf("Stochastic multiplication result for %f * %f = %f\n", a, b, result);
    printf("MAE: %f\n", abs(a * b - result));

    return 0;
}