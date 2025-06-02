#include <stdio.h>
#include "./llmc/SC.h"

int main() {
    // Example input values (outside [0,1), assuming a known range, e.g., [-10,10])
    // float a = -0.125765;
    // float b = 0.012414;

    for(int i = 1; i < 10; i++)
        printf("%f\n", generate_halton_number(i, 3));

    float a = 0.130907;
    float b = 0.108813;
    float result;

    // Define the expected input range for normalization.
    // For example, if we expect x ∈ [-10, 10]:
    float max_abs_val = 1;

    // Call the top-level function for stochastic multiplication.
    // It normalizes the inputs, converts them to fixed point, generates stochastic bitstreams,
    // performs bitwise AND (multiplication), and converts the result back to a float.
    float halton_sequence_base_2[NUM_WIDTH][SN_UNIT];
    float halton_sequence_base_3[NUM_WIDTH][SN_UNIT];
    init_halton(halton_sequence_base_2, 2);
    init_halton(halton_sequence_base_3, 3);
    result = SC_mult(a, b, max_abs_val, halton_sequence_base_2, halton_sequence_base_3);

    // Print the result. Note: Due to the stochastic nature, the result is an approximation.
    printf("binary multiplication result for %f * %f = %f\n", a, b, a * b);
    printf("Stochastic multiplication result for %f * %f = %f\n", a, b, result);
    printf("MAE: %f\n", abs(a * b - result));

    return 0;
}