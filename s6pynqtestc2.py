import numpy as np
from pynq import Overlay, allocate
import time

# bit dan hwh
overlay = Overlay('paket/mc2h-wrapped/mc2h.bit')
dma = overlay.axi_dma_0
quantization_scale = 1 << 10 # 1024

def test_load():
    x2 = np.load('paket/mc2h-wrapped/X_test_main.npy').astype(np.float32) # shape (2, 32, 32, 3)
    y = np.load('paket/mc2h-wrapped/Y_test_main.npy').astype(np.int32)
    xq = np.clip(np.rint(x2 * quantization_scale), -32768, 32767).astype(np.int16)
    return xq,y


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

def unpack_out_u32_to_two_q6_10(y_word):
    y0 = np.int16(y_word & 0xFFFF)
    y1 = np.int16((y_word >> 16) & 0xFFFF)
    return np.array([y0, y1], dtype=np.int16) / quantization_scale


def main():
    X,Y = test_load()
    hasil = []
    total_time = 0
    rerata_samples = []
    # n=(Z⋅σ​)/E
    # Z=1.96 (95% confidence)
    # σ=0.013 (stddev) dari percobaan dengan n=30
    # E=0.01 (margin of error, sekitar 1persen rerata dari pengukuran dengan n=30)
    # n=6.49 
    # n=7
    resampling = 7
    n = Y.shape[0]
    for resample in range(resampling):
        sample_total_time = 0
        benar = 0
        for testcase in range(n):
            in_words = pack_rgb16_to_u64(X[testcase])
            in_buf = allocate(shape=(in_words.size,), dtype=np.uint64)
            out_buf = allocate(shape=(1,), dtype=np.uint32)

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

            probs = unpack_out_u32_to_two_q6_10(int(out_buf[0]))
            # print(f"Testcase {testcase}: infer_time={infer_time*1e3:.3f} ms | DMA in={waktu_dma_in*1e3:.3f} ms | DMA out={waktu_dma_out*1e3:.3f} ms")
            label = np.argmax(probs)
            hasil.append(label)
            if label == Y[testcase]:
                benar += 1

            in_buf.freebuffer()
            out_buf.freebuffer()
            
        # print(f"hasil: {hasil}")
        rerata_sample = sample_total_time / n*1e3
        akurasi_sample = benar / n * 100
        print(f"{resample} Rata-rata latensi total: {rerata_sample:.3f} ms")

        rerata_samples.append(rerata_sample)
        total_time += rerata_sample
    
    rerata_total = total_time / (resampling)
    stddev = np.std(rerata_samples)
    print(f"Akurasi: {benar}/{n} = {akurasi_sample:.2f} %")
    print(f"Rata-rata latensi total dari {resampling} kali pengujian: {rerata_total:.3f} ms")
    print(f"Standar deviasi latensi total: {stddev:.3f} ms")

main()