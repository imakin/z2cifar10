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

// static void core_stream_to_axis64(hls::stream<result_t> &core_out, hls::stream<axis64_t> &out64) {
// #pragma HLS INLINE off
//     // Core now produces 10-class output (result_t has 10 elements
//     result_t res = core_out.read();

//     // Pack 10x26 bits = 260 bits into 5x64-bit words (last 60 bits unused)
//     ap_uint<260> packed = 0;
//     for (int i = 0; i < 10; ++i) {
//         // Sign-extend to 26 bits, then insert into packed
//         ap_int<26> val = res[i].range(25, 0); // ap_fixed<26,13>
//         packed.range((i+1)*26-1, i*26) = val;
//     }

//     // Send as 5 AXIS 64-bit beats
//     for (int i = 0; i < 5; ++i) {
//         axis64_t w;
//         w.data = 0;
//         w.keep = 0xFF;
//         w.strb = 0xFF;
//         w.last = (i == 4) ? 1 : 0;
//         ap_uint<64> d = packed.range((i+1)*64-1, i*64);
//         w.data = d;
//         out64.write(w);
//     }
// }




static void core_stream_to_axis64(hls::stream<result_t> &core_out, hls::stream<axis64_t> &out64) {
#pragma HLS INLINE off
    // Output: 10x16 bit (ap_fixed<16,6>) = 160 bit, send as 3x64 bit in 3 beats
    result_t res = core_out.read();

    for (int beat=0; beat<3; beat++){
        axis64_t w;
        w.data = 0;
        w.data.range(15,0)  = res[beat*4 +0].range(15,0);
        w.data.range(31,16) = res[beat*4 +1].range(15,0);
        if ((beat*4 +2) < 10) {
            w.data.range(47,32) = res[beat*4 +2].range(15,0);
            w.data.range(63,48) = res[beat*4 +3].range(15,0);
        }
        w.keep = 0xFF;
        w.strb = 0xFF;
        w.last = (beat >= 2) ? 1 : 0;
        out64.write(w);
    }
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
