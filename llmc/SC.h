#ifndef SC_H
#define SC_H

#include <ap_fixed.h>
#include <ap_int.h>

#define SN_LEN 1024 // Length of the stochastic bitstream

extern ap_uint<24> g_lfsr_state = 0xACE1;

// Normalize a float to [0,1) given a known min and max range.
float normalize_clip(float x, float max_val);

// Denormalize
float denormalize(float x, float max_val);

// Convert a normalized float (assumed in [0,1)) to fixed-point in Q0.24.
ap_uint<24> float_to_fixed24(float x_norm);

// 24-bit LFSR for pseudo-random number generation.
ap_uint<24> lfsr24(ap_uint<24> state);

// Generate a stochastic bitstream from a fixed-point threshold using a given seed.
ap_uint<SN_LEN> gen_SN(ap_uint<SN_LEN> stream, ap_uint<24> seed);

// Multiply two stochastic bitstreams (element-wise AND).
// void SC_Mul(ap_uint<1> stream1[SN_LEN], ap_uint<1> stream2[SN_LEN], ap_uint<1> out_stream[SN_LEN]);

// Convert a stochastic bitstream back to a float by averaging the bits.
float SN_to_float(ap_uint<SN_LEN> stream);

// Top-level function to perform stochastic multiplication of two floats.
// The inputs a and b are normalized using [min_val, max_val] and then converted into
// a stochastic bitstream. Their multiplication is performed via bitwise AND, and the
// result is averaged back into a float.
float SC_mult(float a, float b, float max_val, ap_uint<24> seed1, ap_uint<24> seed2
);

#endif // SC_H