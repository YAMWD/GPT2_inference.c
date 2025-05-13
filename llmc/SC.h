#ifndef SC_H
#define SC_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// Normalize a float to [0,1) given a known min and max range.
float normalize_clip(float x, float max_val);

// Denormalize
float denormalize(float x, float max_val);

// Convert a normalized float (assumed in [0,1)) to fixed-point in Q0.24.
ap_uint<24> float_to_fixed24(float x_norm);

// 24-bit LFSR for pseudo-random number generation.
ap_uint<24> next_lfsr24(ap_uint<24> g_lfsr_state);

// Generate a stochastic bitstream from a fixed-point threshold using a given seed.
void gen_SN(float p, hls::stream<ap_uint<1>> &stream, ap_uint<24> lfsr_state, int SN_LEN);

// Convert a stochastic bitstream back to a float by averaging the bits.
float SN_to_float(hls::stream<ap_uint<1>> &stream, int SN_LEN);

void SC_Mul(hls::stream<ap_uint<1>> &stream_a, hls::stream<ap_uint<1>> &stream_b, hls::stream<ap_uint<1>> &stream_out, int SN_LEN);

// Top-level function to perform stochastic multiplication of two floats.
// The inputs a and b are normalized using [min_val, max_val] and then converted into
// a stochastic bitstream. Their multiplication is performed via bitwise AND, and the
// result is averaged back into a float.
float SC_mult(float a, float b, float max_val, int SN_LEN);

#endif // SC_H