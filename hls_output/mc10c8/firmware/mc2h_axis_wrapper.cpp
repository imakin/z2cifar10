#include "mc2h_axis_wrapper.h"
// #include "parameters.h" // not required here

// Image dimensions known from defines.h: 32x32x3
static const int IMG_H = 32;
static const int IMG_W = 32;
static const int IMG_PIXELS = IMG_H * IMG_W; // 1024

static void axis64_to_core_stream(hls::stream<axis64_t> &in64, hls::stream<input_t> &core_in) {
#pragma HLS INLINE off
    for (int i = 0; i < IMG_PIXELS; ++i) {
#pragma HLS PIPELINE II=1
        axis64_t w = in64.read();
        ap_uint<64> d = w.data;
        // Unpack 3x int16 (R,G,B) from bits [15:0], [31:16], [47:32]
        ap_uint<16> r_u = d.range(15, 0);
        ap_uint<16> g_u = d.range(31, 16);
        ap_uint<16> b_u = d.range(47, 32);

        input_t pix;
        // Bit-accurate reinterpretation: assign raw bits into ap_fixed storage
        pix[0].range(15, 0) = r_u;
        pix[1].range(15, 0) = g_u;
        pix[2].range(15, 0) = b_u;
        core_in.write(pix);
    }
}

static void core_stream_to_axis64(hls::stream<result_t> &core_out, hls::stream<axis64_t> &out64) {
#pragma HLS INLINE off
    // Core now produces 10-class output (result_t has 10 elements, each 16 bit)
    result_t res = core_out.read();

    // Beat 1: res[0..3] (4x16=64 bit)
    axis64_t w1;
    w1.data = 0;
    w1.keep = 0xFF; // all 8 bytes valid
    w1.strb = 0xFF;
    w1.last = 0;
    ap_uint<64> d1 = 0;
    d1.range(15, 0)   = res[0].range(15, 0);
    d1.range(31, 16)  = res[1].range(15, 0);
    d1.range(47, 32)  = res[2].range(15, 0);
    d1.range(63, 48)  = res[3].range(15, 0);
    w1.data = d1;
    out64.write(w1);

    // Beat 2: res[4..7]
    axis64_t w2;
    w2.data = 0;
    w2.keep = 0xFF;
    w2.strb = 0xFF;
    w2.last = 0;
    ap_uint<64> d2 = 0;
    d2.range(15, 0)   = res[4].range(15, 0);
    d2.range(31, 16)  = res[5].range(15, 0);
    d2.range(47, 32)  = res[6].range(15, 0);
    d2.range(63, 48)  = res[7].range(15, 0);
    w2.data = d2;
    out64.write(w2);

    // Beat 3: res[8..9] (2x16=32 bit), rest zero
    axis64_t w3;
    w3.data = 0;
    w3.keep = 0x0F; // lower 4 bytes valid (2x16=32 bit)
    w3.strb = 0x0F;
    w3.last = 1;
    ap_uint<64> d3 = 0;
    d3.range(15, 0)   = res[8].range(15, 0);
    d3.range(31, 16)  = res[9].range(15, 0);
    // d3.range(63, 32) = 0; // already zero
    w3.data = d3;
    out64.write(w3);
}

void mc2h_axis64(hls::stream<axis64_t> &in64, hls::stream<axis64_t> &out64) {
#pragma HLS INTERFACE axis port=in64
#pragma HLS INTERFACE axis port=out64
#pragma HLS INTERFACE ap_ctrl_none port=return

    hls::stream<input_t> core_in("core_in");
    hls::stream<result_t> core_out("core_out");
#pragma HLS STREAM variable=core_in depth=1024
#pragma HLS STREAM variable=core_out depth=2

    // Continuous frame processing in HW, single frame in C-simulation
#ifdef __SYNTHESIS__
    while (1) {
#pragma HLS DATAFLOW
        axis64_to_core_stream(in64, core_in);
        mc10c8(core_in, core_out);
        core_stream_to_axis64(core_out, out64);
    }
#else
#pragma HLS DATAFLOW
    axis64_to_core_stream(in64, core_in);
    mc10c8(core_in, core_out);
    core_stream_to_axis64(core_out, out64);
#endif
}
