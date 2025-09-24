#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "mc2h_axis_wrapper.h"

// Minimal NPY reader for C-sim (float32 or int16 arrays)
// Supports little-endian C-order arrays.
struct NpyArray {
    std::vector<unsigned char> data;
    std::vector<size_t> shape;
    std::string descr; // e.g., "<f4" for float32, "<i2" for int16
};

static NpyArray load_npy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("Cannot open npy file: " + path); }
    char magic[6]; f.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("Bad NPY magic");
    char v_major, v_minor; f.read(&v_major, 1); f.read(&v_minor, 1);
    uint16_t header_len_le; f.read(reinterpret_cast<char*>(&header_len_le), 2);
    size_t header_len = header_len_le; // little-endian
    std::string header(header_len, '\0'); f.read(&header[0], header_len);
    // parse descr
    auto find_between = [&](const std::string &s, const char *a, const char *b){
        auto i = s.find(a); if (i==std::string::npos) return std::string(); i += std::strlen(a);
        auto j = s.find(b, i); if (j==std::string::npos) return std::string();
        return s.substr(i, j-i);
    };
    std::string descr = find_between(header, "'descr': '", "'");
    std::string shape_s = find_between(header, "'shape': (", ")");
    std::vector<size_t> shape;
    {
        size_t pos=0; while (pos < shape_s.size()) {
            while (pos<shape_s.size() && (shape_s[pos]==' '||shape_s[pos]==',')) ++pos;
            size_t start=pos; while (pos<shape_s.size() && isdigit(shape_s[pos])) ++pos;
            if (start<pos) shape.push_back(std::stoul(shape_s.substr(start, pos-start)));
            while (pos<shape_s.size() && (shape_s[pos]==' '||shape_s[pos]==',')) ++pos;
        }
    }
    // read data
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return {std::move(buf), std::move(shape), std::move(descr)};
}

// Quantize float32 [H,W,3] in range roughly [-2,2] to Q6.10 int16 raw bits
static inline int16_t float_to_q6_10(float x) {
    float y = std::roundf(x * 1024.0f); // 2^10
    if (y >  32767.f) y =  32767.f;
    if (y < -32768.f) y = -32768.f;
    return static_cast<int16_t>(y);
}

int main(int argc, char **argv) {
    try {
        // Input can be either:
        //  - float32 npy with shape (32,32,3) or (N,32,32,3)
        //  - int16  npy with shape (32,32,3) or (N,32,32,3) already Q6.10 raw bits
        std::string path = (argc > 1) ? argv[1] : "X_test2.npy";
        auto arr = load_npy(path);
        if (arr.shape.size() != 3 && arr.shape.size() != 4)
            throw std::runtime_error("Expected shape (H,W,3) or (N,H,W,3)");
        size_t N = (arr.shape.size()==4) ? arr.shape[0] : 1;
        size_t H = arr.shape[arr.shape.size()-3];
        size_t W = arr.shape[arr.shape.size()-2];
        size_t C = arr.shape[arr.shape.size()-1];
        if (H!=32 || W!=32 || C!=3) throw std::runtime_error("Expected (32,32,3)");

        size_t elems_per_image = H*W*C; // 3072
        size_t bytes_per_elem = (arr.descr=="<f4") ? 4 : ((arr.descr=="<i2") ? 2 : 0);
        if (bytes_per_elem==0) throw std::runtime_error("Unsupported dtype: " + arr.descr);

        // Process first image only in this TB
        const unsigned char *base = arr.data.data();
        if (arr.shape.size()==4) {
            // offset 0 already points to first image
        }

        hls::stream<axis64_t> s_in("s_in");
        hls::stream<axis64_t> s_out("s_out");

        // Build 1024 beats: one per pixel in row-major, each beat packs R,G,B into 16-bit lanes
        for (size_t i=0;i<32*32;i++) {
            size_t off = i*3;
            float rf=0,gf=0,bf=0;
            int16_t r16=0,g16=0,b16=0;
            if (arr.descr=="<f4") {
                const float *fptr = reinterpret_cast<const float*>(base);
                rf = fptr[off+0]; gf = fptr[off+1]; bf = fptr[off+2];
                r16 = float_to_q6_10(rf);
                g16 = float_to_q6_10(gf);
                b16 = float_to_q6_10(bf);
            } else { // <i2
                const int16_t *iptr = reinterpret_cast<const int16_t*>(base);
                r16 = iptr[off+0]; g16 = iptr[off+1]; b16 = iptr[off+2];
            }
            ap_uint<64> d = 0;
            d.range(15,0)  = (ap_uint<16>)r16;
            d.range(31,16) = (ap_uint<16>)g16;
            d.range(47,32) = (ap_uint<16>)b16;
            axis64_t w; w.data = d; w.keep = 0xFF; w.strb = 0xFF; w.last = 0;
            s_in.write(w);
        }

        // Call DUT
        mc2h_axis64(s_in, s_out);

        // Output now consists of 3 beats (10 label, 16 bit each)
        int16_t y[10] = {0};
        for (int i = 0; i < 3; ++i) {
            if (s_out.empty()) {
                std::fprintf(stderr, "ERROR: Output beat %d missing\n", i);
                return 1;
            }
            axis64_t o = s_out.read();
            ap_uint<64> od = o.data;
            if (i == 0) {
                y[0] = (int16_t)od.range(15,0);
                y[1] = (int16_t)od.range(31,16);
                y[2] = (int16_t)od.range(47,32);
                y[3] = (int16_t)od.range(63,48);
            } else if (i == 1) {
                y[4] = (int16_t)od.range(15,0);
                y[5] = (int16_t)od.range(31,16);
                y[6] = (int16_t)od.range(47,32);
                y[7] = (int16_t)od.range(63,48);
            } else if (i == 2) {
                y[8] = (int16_t)od.range(15,0);
                y[9] = (int16_t)od.range(31,16);
                // upper bits are zero
            }
            std::printf("Output beat %d: data=0x%016llx (TLAST=%u)\n", i, (unsigned long long)od, (unsigned)o.last);
        }
        std::printf("Output Q6.10 labels: ");
        for (int i = 0; i < 10; ++i) {
            std::printf("%d ", (int)y[i]);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 2;
    }
}
