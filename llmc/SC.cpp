#include "SC.h"
#include <stdio.h>
#include <math.h>

ap_uint<24> g_lfsr_state = 0xACE1;

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
ap_uint<24> next_lfsr24() 
{
    #pragma HLS inline
    bool new_bit = g_lfsr_state[23] ^ g_lfsr_state[22] ^ g_lfsr_state[20] ^ g_lfsr_state[19];
    g_lfsr_state = (g_lfsr_state << 1) | new_bit;
    return g_lfsr_state; 
}

// Generate a stochastic bitstream from the fixed-point threshold.
// Each bit in the stream is 1 if the pseudo-random number is less than fixed_val.
// void gen_SN(ap_uint<24> fixed_val, ap_uint<1> stream[SN_LEN], ap_uint<24> seed) 
// {
//     #pragma HLS inline off
//     ap_uint<24> state = seed;
//     for (int i = 0; i < SN_LEN; i++) {
//     #pragma HLS pipeline II=1
//         state = lfsr24(state);
//         stream[i] = (state < fixed_val) ? 1 : 0;
//     }
// }

ap_uint<SN_LEN> gen_SN(float p) {
    ap_uint<SN_LEN> bitstream = 0;
    ap_uint<24> threshold = float_to_fixed24(p);

    // Fill each bit of the packed stream
    for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
    #pragma HLS unroll factor=8
        bitstream[i] = (next_lfsr24() < threshold) ? 1 : 0;
    }

    return bitstream;
}

// Perform stochastic multiplication via bitwise XNOR on two bitstreams.
// void SC_Mul(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN], ap_uint<1> out_stream[SN_LEN]) 
// {
//     for (int i = 0; i < SN_LEN; i++) {
//     #pragma HLS pipeline II=1
//         out_stream[i] = !(stream1[i] ^ stream2[i]);
//     }
// }

// Average the bits in a stochastic bitstream to recover an approximate float value.
float SN_to_float(ap_uint<SN_LEN> stream) 
{
    int sum = 0;
    for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
    #pragma HLS unroll factor=8
        sum += stream[i];
    }
    // printf("%d\n", sum);
    return (float)(sum + sum) / SN_LEN - 1;
}

// Top-level function for stochastic multiplication.
// This function normalizes the inputs, converts them to fixed-point, generates the corresponding
// stochastic bitstreams, performs a bitwise AND for multiplication, and then converts the result back to float.
float SC_mult(float a, float b, float max_val) 
{
    #pragma HLS INTERFACE s_axilite port=a      bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=b      bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=max_val bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    // Normalize inputs to [0,1) based on the expected range.
    float normed_a = normalize_clip(a, max_val);
    float normed_b = normalize_clip(b, max_val);

    // printf("normed a b: %f %f\n", normed_a, normed_b);

    // Generate stochastic bitstreams for both operands.
    ap_uint<SN_LEN> stream_a = gen_SN(normed_a);
    ap_uint<SN_LEN> stream_b = gen_SN(normed_b);

    // bipolar SC mult is done by an XNOR gate
    ap_uint<SN_LEN> stream_out = ~(stream_a ^ stream_b);

    // float float_a = SN_to_float(stream_a);
    // float float_b = SN_to_float(stream_b);
    // printf("%f %f\n", float_a, float_b);

    // Convert the result bitstream back to a float.
    return denormalize(SN_to_float(stream_out), max_val);
}