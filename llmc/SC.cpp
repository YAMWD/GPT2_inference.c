#include "SC.h"
#include <stdio.h>
#include <math.h>

// Normalize x to the [0,1) range using provided min and max values.
float normalize_clip(float x, float max_val) 
{
    // normed_x = x / max_val
    // prob = (normed_x + 1) / 2
    float x_norm = x / (max_val + max_val) + 0.5;
    if (x_norm >= 1.0f) x_norm = 0.999999f;
    if (x_norm < 0.0f)  x_norm = 0.0f;
    return x_norm;
}

// Denormalize x
float denormalize(float x, float max_val) 
{
    // x * max_val ^ 2
    return x * max_val * max_val;
}

// Convert a normalized float (in [0,1)) to a fixed-point representation (Q0.24).
ap_uint<24> float_to_fixed24(float x_norm) 
{
    // Using ap_fixed with 24 total bits and 0 integer bits yields 24 fractional bits.
    ap_fixed<24,0> fixed_val = x_norm;
    return fixed_val.range(23,0);
}

// A simple 24-bit LFSR; adjust feedback taps as needed.
ap_uint<24> lfsr24(ap_uint<24> state) 
{
    #pragma HLS inline
    bool new_bit = state[23] ^ state[22] ^ state[20] ^ state[19];
    return (state << 1) | new_bit;
}

// Generate a stochastic bitstream from the fixed-point threshold.
// Each bit in the stream is 1 if the pseudo-random number is less than fixed_val.
void gen_SN(ap_uint<24> fixed_val, ap_uint<1> stream[SN_LEN], ap_uint<24> seed) 
{
    #pragma HLS inline off
    ap_uint<24> state = seed;
    for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
        state = lfsr24(state);
        stream[i] = (state < fixed_val) ? 1 : 0;
    }
}

// Perform stochastic multiplication via bitwise XNOR on two bitstreams.
void SC_Mul(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN], ap_uint<1> out_stream[SN_LEN]) 
{
    for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
        out_stream[i] = !(stream1[i] ^ stream2[i]);
    }
}

// Average the bits in a stochastic bitstream to recover an approximate float value.
float SN_to_float(ap_uint<1> stream[SN_LEN]) 
{
    int sum = 0;
    for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
        sum += stream[i];
    }
    // printf("%d\n", sum);
    return (float)(sum + sum) / SN_LEN - 1;
}

// Top-level function for stochastic multiplication.
// This function normalizes the inputs, converts them to fixed-point, generates the corresponding
// stochastic bitstreams, performs a bitwise AND for multiplication, and then converts the result back to float.
void SC_mult(
    float a, float b, float *result, float max_val, 
    ap_uint<24> seed1, ap_uint<24> seed2) 
{
    #pragma HLS INTERFACE s_axilite port=a      bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=b      bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=result bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=max_val bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=seed1   bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=seed2   bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    // Normalize inputs to [0,1) based on the expected range.
    float a_norm = normalize_clip(a, max_val);
    float b_norm = normalize_clip(b, max_val);

    // printf("normed a b: %f %f\n", a_norm, b_norm);

    // Convert normalized values to fixed-point Q0.24 representation.
    ap_uint<24> fixed_a = float_to_fixed24(a_norm);
    ap_uint<24> fixed_b = float_to_fixed24(b_norm);

    // Generate stochastic bitstreams for both operands.
    ap_uint<1> stream_a[SN_LEN], stream_b[SN_LEN], stream_out[SN_LEN];
    gen_SN(fixed_a, stream_a, seed1);
    gen_SN(fixed_b, stream_b, seed2);

    float float_a = SN_to_float(stream_a);
    float float_b = SN_to_float(stream_b);

    // printf("%f %f\n", float_a, float_b);

    // Multiply the stochastic bitstreams (bitwise AND for unipolar representation).
    SC_Mul(stream_a, stream_b, stream_out);

    // Convert the result bitstream back to a float.
    *result = denormalize(SN_to_float(stream_out), max_val);
}