#ifndef MC2H_AXIS_WRAPPER_H_
#define MC2H_AXIS_WRAPPER_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

#include "defines.h"
#include "mc2hc6.h"

// 64-bit AXIS type: data/keep/strb/last supported; no user/id/dest fields used
typedef ap_axiu<64, 0, 0, 0> axis64_t;

// Top-level wrapper exposing 64-bit AXIS on both input and output.
// - Input: one 64-bit beat per pixel; lower 48 bits carry 3x int16 (R,G,B) in that order.
// - Output: one 64-bit beat containing 2x int16 in lower 32 bits; keep/strb=0x0F, upper bytes zero.
void mc2h_axis64(
    hls::stream<axis64_t> &in64,
    hls::stream<axis64_t> &out64
);

#endif
