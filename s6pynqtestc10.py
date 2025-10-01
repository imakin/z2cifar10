import numpy as np
from pynq import Overlay, allocate
import time

# bit dan hwh
overlay = Overlay('paket/mc10/mc10c8.bit')
# dma = overlay.axi_dma_0

# Desain HLS menggunakan tipe input ap_fixed<16,6> => 6 bit integer (dengan sign) + 10 bit fraksi (Q6.10)
# Jadi faktor kuantisasi yang benar adalah 2^10 = 1024, BUKAN 2^15. Penggunaan 2^15 (Q1.15) sebelumnya
# menyebabkan hampir semua nilai menjenuh (saturate) ke +/- 31.999 pada hardware sehingga
# output klasifikasi menjadi salah dan berulang.
quantization_scale = 1 << 10  # Q6.10

def test_load():
    x2 = np.load('paket/mc10/c10_X_test_main.npy').astype(np.float32) # shape (2, 32, 32, 3)
    y  = np.load('paket/mc10/c10_Y_test_main.npy').astype(np.int32)
    xq = np.clip(np.rint(x2 * quantization_scale), -32768, 32767).astype(np.int16)
    return xq,y
X,Y = test_load()



def pack_rgb16_to_u64(img_q16):
    # img_q16: (H, W, 3) int16
    r = img_q16[..., 0].astype(np.uint16)
    g = img_q16[..., 1].astype(np.uint16)
    b = img_q16[..., 2].astype(np.uint16)
    word = (
        r.astype(np.uint64)
        | (g.astype(np.uint64) << 16)
        | (b.astype(np.uint64) << 32)
    )
    return word.reshape(-1).astype(np.uint64) # length = H*W






# Fungsi unpack untuk 10 label dari 3 beat AXIS 64-bit (ap_fixed<16,6>)
def unpack_out_u64_to_10labels(y_words):
    # y_words: array 3xuint64
    packed = int(y_words[0]) | (int(y_words[1]) << 64) | (int(y_words[2]) << 128)
    arr = np.zeros(10, dtype=np.int16)
    for i in range(10):
        val = (packed >> (i*16)) & 0xFFFF
        # sign-extend 16 bit to int16
        if val & 0x8000:
            arr[i] = val - 0x10000
        else:
            arr[i] = val
    return arr.astype(np.float32) / (1 << 10)  # <16,6>: 10 bit fractional

def unpack_out_u64_to_10labels2(y_words):
    arr = np.zeros(10, dtype=np.int16)
    y0 = int(y_words[0])
    y1 = int(y_words[1])
    y2 = int(y_words[2])
    arr[0] = np.int16( y0 & 0xFFFF)
    arr[1] = np.int16((y0 >> 16) & 0xFFFF)
    arr[2] = np.int16((y0 >> 32) & 0xFFFF)
    arr[3] = np.int16((y0 >> 48) & 0xFFFF)
    arr[4] = np.int16( y1 & 0xFFFF)
    arr[5] = np.int16((y1 >> 16) & 0xFFFF)
    arr[6] = np.int16((y1 >> 32) & 0xFFFF)
    arr[7] = np.int16((y1 >> 48) & 0xFFFF)
    arr[8] = np.int16( y2 & 0xFFFF)
    arr[9] = np.int16((y2 >> 16) & 0xFFFF)
    for i in range(10):
        if arr[i] & 0x8000:
            arr[i] = arr[i] - 0x10000
    return arr.astype(np.float32) / (1 << 10)  # <16


def main(dma, resampling=8):
    hasil = []
    total_time = 0
    rerata_samples = []
    n = Y.shape[0]
    for resample in range(resampling):
        sample_total_time = 0
        benar = 0
        for testcase in range(n):
            in_words = pack_rgb16_to_u64(X[testcase])
            in_buf = allocate(shape=(in_words.size,), dtype=np.uint64)

            out_buf = allocate(shape=(3,), dtype=np.uint64)  # 3x64 bit output

            np.copyto(in_buf, in_words)

            t0 = time.perf_counter()
            dma.sendchannel.transfer(in_buf)
            dma.sendchannel.wait()
            t1 = time.perf_counter()
            dma.recvchannel.transfer(out_buf)
            dma.recvchannel.wait()
            t2 = time.perf_counter()

            waktu_dma_in = t1 - t0
            waktu_dma_out = t2 - t1
            infer_time = t2 - t0
            sample_total_time += infer_time

            try:
                probs = unpack_out_u64_to_10labels(out_buf)
            except TypeError as e:
                print(out_buf)
                print(type(out_buf[0]))
                raise e
            label = np.argmax(probs)
            hasil.append(label)
            if label == Y[testcase]:
                benar += 1

            in_buf.freebuffer()
            out_buf.freebuffer()
            
        # print(f"hasil: {hasil}")
        rerata_sample = sample_total_time / n*1e3
        akurasi_sample = benar / n * 100
        print(f"{resample} Rata-rata latensi total: {rerata_sample:.3f} ms. akurasi sample: {akurasi_sample}")

        rerata_samples.append(rerata_sample)
        total_time += rerata_sample
    
    rerata_total = total_time / (resampling)
    stddev = np.std(rerata_samples)
    print(f"Akurasi: {benar}/{n} = {akurasi_sample:.2f} %")
    print(f"Rata-rata latensi total dari {resampling} kali pengujian: {rerata_total:.3f} ms")
    print(f"Standar deviasi latensi total: {stddev:.3f} ms")
