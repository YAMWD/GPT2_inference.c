#include "SC.h"
#include <stdio.h>
#include <math.h>

// Normalize x to the [0,1) range using provided min and max values.
float normalize_clip(float x) 
{
    #pragma HLS inline 
    // normed_x = x / max_val
    // prob = (normed_x + 1) / 2

    float x_norm = (x + 1) / 2;
    if (x_norm >= 1.0f) x_norm = 0.999999f;
    if (x_norm < 0.0f)  x_norm = 0.0f;
    return x_norm;
}

// Denormalize x
float denormalize(float x, float max_val) 
{
    #pragma HLS inline 
    // x * max_val ^ 2
    return x * max_val * max_val;
}

// Convert a normalized float (in [0,1)) to a fixed-point representation (Q0.24).
ap_uint<24> float_to_fixed24(float x_norm) 
{
    #pragma HLS inline 
    // Using ap_fixed with 24 total bits and 0 integer bits yields 24 fractional bits.
    ap_fixed<24,0> fixed_val = x_norm;
    return fixed_val.range(23,0);
}

// A simple 24-bit LFSR; adjust feedback taps as needed.
ap_uint<24> next_lfsr24(ap_uint<24> g_lfsr_state) 
{
    #pragma HLS inline 
    bool new_bit = g_lfsr_state[23] ^ g_lfsr_state[22] ^ g_lfsr_state[20] ^ g_lfsr_state[19];
    g_lfsr_state = (g_lfsr_state << 1) | new_bit;
    return g_lfsr_state; 
}

void gen_SN(float p, ap_uint<1> stream[SN_LEN], ap_uint<24> lfsr_state) 
{
    #pragma HLS inline

    ap_uint<24> threshold = float_to_fixed24(p);

    gen_SN: for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
        lfsr_state = next_lfsr24(lfsr_state);
        stream[i] = (lfsr_state < threshold) ? 1 : 0;
    }
}

float SN_to_float(ap_uint<1> stream[SN_LEN]) 
{
    #pragma HLS inline

    int64_t sum = 0;
    SN_to_BN: for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS pipeline II=1
        if (stream[i] == 1)
            sum = -~sum;
        else
            sum = ~-sum;
    }

    return (float)sum / SN_LEN;
}

void SC_Mul(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN], ap_uint<1> out_stream[SN_LEN]) 
{
    #pragma HLS ARRAY_PARTITION variable=stream1 complete
    #pragma HLS ARRAY_PARTITION variable=stream2 complete
    #pragma HLS ARRAY_PARTITION variable=out_stream complete

    SC_MUL: for (int i = 0; i < SN_LEN; i++) {
    #pragma HLS unroll
        out_stream[i] = !(stream1[i] ^ stream2[i]);
    }
}

#ifndef __SYNTHESIS__
float compute_PC(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN])
{
    int N[2][2] = {0};
    for (int i = 0; i < SN_LEN; i++)
        N[stream1[i]][stream2[i]]++;

    if (N[1][1] * N[0][0] > N[1][0] * N[0][1])
        return (float)(N[1][1] * N[0][0] - N[1][0] * N[0][1]) / (SN_LEN * std::min(N[1][1] + N[1][0], N[1][1] + N[0][1]) - (N[1][1] + N[1][0]) * (N[1][1] + N[0][1]));
    else    
        return (float)(N[1][1] * N[0][0] - N[1][0] * N[0][1]) / ((N[1][1] + N[1][0]) * (N[1][1] + N[0][1]) - SN_LEN * std::max(N[1][1] - N[0][0], 0));
}
#endif

// Top-level function for stochastic multiplication.
// This function normalizes the inputs, converts them to fixed-point, generates the corresponding
// stochastic bitstreams, performs a bitwise AND for multiplication, and then converts the result back to float.
float SC_mult(float a, float b) 
{
    #pragma HLS INTERFACE s_axilite port=a       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=b       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    #pragma HLS inline

    // Normalize inputs to [0,1) based on the expected range.
    float normed_a = normalize_clip(a);
    float normed_b = normalize_clip(b);

    // printf("normed a b: %f %f\n", normed_a, normed_b);

    ap_uint<24> lfsr_state_1 = 0xACE1;
    ap_uint<24> lfsr_state_2 = 0xBCE1;

    // Generate stochastic bitstreams for both operands.
    ap_uint<1> stream_a[SN_LEN];
    ap_uint<1> stream_b[SN_LEN];
    ap_uint<1> stream_out[SN_LEN];

    gen_SN(normed_a, stream_a, lfsr_state_1);
    gen_SN(normed_b, stream_b, lfsr_state_2);

    // printf("pearson correlation %f\n ", compute_PC(stream_a, stream_b));

    // printf("%f %f\n\n", SN_to_float(stream_a), SN_to_float(stream_b));
    // bipolar SC mult is done by an XNOR gate
    SC_Mul(stream_a, stream_b, stream_out);

    // printf("%f\n", SN_to_float(stream_out));

    // Convert the result bitstream back to a float.
    return SN_to_float(stream_out);
    // return SN_to_float(stream_out);

}