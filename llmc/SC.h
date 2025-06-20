#ifndef SC_H
#define SC_H

#include <ap_fixed.h>
#include <ap_int.h>

#define SN_LEN 1024 // stochastic bitstream length
#define NUM_WIDTH 1024
#define SN_UNIT (SN_LEN / NUM_WIDTH)

typedef ap_uint<SN_UNIT> SN;

// Normalize a float to [0,1) given a known min and max range.
float normalize_clip(float x, float max_val);

// Denormalize
float denormalize(float x, float max_val);

// Convert a normalized float (assumed in [0,1)) to fixed-point in Q0.24.
ap_uint<24> float_to_fixed24(float x_norm);

// 24-bit LFSR for pseudo-random number generation.
ap_uint<24> next_lfsr24(ap_uint<24> g_lfsr_state);

// Generate a stochastic bitstream from a fixed-point threshold using a given seed.
// SN gen_SN(float p, ap_uint<24> lfsr_state);

// Convert a stochastic bitstream back to a float by averaging the bits.
float SN_to_float(SN stream[NUM_WIDTH]);

void gen_SN(float p, SN stream[NUM_WIDTH]);

// // Multiply two stochastic bitstreams (element-wise AND).
// void SC_Mul(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN], ap_uint<1> out_stream[SN_LEN]);

// float SN_to_float(ap_uint<1> stream[SN_LEN]);

// Top-level function to perform stochastic multiplication of two floats.
// The inputs a and b are normalized using [min_val, max_val] and then converted into
// a stochastic bitstream. Their multiplication is performed via bitwise AND, and the
// result is averaged back into a float.
float SC_mult(float a, float b, float max_val);

#endif // SC_H