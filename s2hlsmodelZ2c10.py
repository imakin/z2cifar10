class Abstract(object):pass

from os import uname
from os import listdir
from os import makedirs
from os.path import realpath, basename
os = Abstract();os.path=Abstract()
os.path.realpath = realpath
os.path.basename = basename

import shutil
import time
from sys import argv
import argparse

import tensorflow as tf
tf.keras.backend.clear_session()  # hindari graph lama tertinggal
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.keras.models import Model
from qkeras import QActivation, QDense, QConv2DBatchnorm
from tensorflow.keras.utils import custom_object_scope
from qkeras.utils import _add_supported_quantized_objects  # ADD
import numpy as np
import hls4ml
from sklearn.metrics import accuracy_score
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from subprocess import check_output as co
from subprocess import Popen as po
def exec(cmd):
    return co(cmd, shell=True).decode('utf8')
def exec2(*args, **kwargs):
    p = po(*args, **kwargs)


def fold_dense_bn(dense_layer, bn_layer, name='folded_dense'):
    """
    Gabung parameter dari layer BatchNormalization ke dalam layer Dense.
    Mengembalikan layer Dense baru yang sudah terlipat.
    """
    if not isinstance(dense_layer, (QDense, Dense)) or not isinstance(bn_layer, BatchNormalization):
        raise ValueError("Layers must be an instance of Dense/QDense and BatchNormalization")

    # Ekstrak parameter dari BatchNormalization
    gamma = bn_layer.get_weights()[0]
    beta = bn_layer.get_weights()[1]
    mean = bn_layer.get_weights()[2]
    variance = bn_layer.get_weights()[3]
    epsilon = bn_layer.epsilon

    # Ekstrak parameter dari Dense
    W_old = dense_layer.get_weights()[0]
    # Jika dense layer tidak punya bias, inisialisasi dengan nol
    b_old = dense_layer.get_weights()[1] if dense_layer.use_bias else np.zeros_like(beta)

    # Hitung bobot dan bias baru
    scale = gamma / np.sqrt(variance + epsilon)
    W_new = W_old * scale
    b_new = (b_old - mean) * scale + beta

    # Buat layer Dense baru dengan konfigurasi yang sama
    # tapi pastikan use_bias=True karena sekarang kita punya bias baru
    config = dense_layer.get_config()
    config['use_bias'] = True
    config['name'] = name
    if isinstance(dense_layer, QDense):
        new_dense_layer = QDense.from_config(config)
    else:
        new_dense_layer =Dense.from_config(config)
    
    # Set bobot baru ke layer yang baru dibuat
    new_dense_layer.build(dense_layer.input_shape)
    new_dense_layer.set_weights([W_new, b_new])
    return new_dense_layer

qco = {}
ret = _add_supported_quantized_objects({})
if isinstance(ret, dict):
    qco.update(ret)                 # versi yang return dict
else:
    _add_supported_quantized_objects(qco)  # versi yang mutasi in-place

custom_objects = {
    **qco,
    'QActivation': QActivation,
    'QDense': QDense,
    'QConv2DBatchnorm': QConv2DBatchnorm,
    'PruneLowMagnitude': PruneLowMagnitude
}

def clone_with_weights(src_model):
    # Clone membuat graph baru yang konsisten; lalu salin bobot per-layer by name
    with custom_object_scope(custom_objects):
        cloned = tf.keras.models.clone_model(src_model)
    name2layer_src = {l.name: l for l in src_model.layers}
    for l in cloned.layers:
        if l.name in name2layer_src:
            w = name2layer_src[l.name].get_weights()
            if w:
                l.set_weights(w)
    return cloned


parser = argparse.ArgumentParser()
parser.add_argument('--sambung', action='store_true') # tidak dipakai lagi.
parser.add_argument('--input', type=str, required=False, default='')
parser.add_argument('--output', type=str, required=False, default='')
parser.add_argument('--minim', action='store_true', help='bikin vitis project minimal')
parser.add_argument('--c10', action='store_true', help='use cifar10 dataset')
parser.add_argument('--skip-profiling', action='store_true', help='skip numerical profiling untuk menghindari hang')
args = parser.parse_args()

input_modelfull = args.input
outputfoldername = args.output

print(f'quick usage untuk 1 model full:')
print(f'`python {os.path.basename(__file__)} --input=[filename model full] --output=[output folder name]`')

# ----- Load model and test data -----
kerases_folder = listdir('keras/')
kerases_folder.sort()
kerases_folder = [f'keras/{f}' for f in kerases_folder]
print("pilih model")
choices = []


print('pilih model FULL: [1/2/3..]')
if input_modelfull:
    model_filenameFULL = input_modelfull
else:
    for i in range(len(kerases_folder)):
        print(f'{kerases_folder[i]}:')
        kerases = listdir(kerases_folder[i])
        kerases = [f'{kerases_folder[i]}/{f}' for f in kerases]
        for j in range(len(kerases)):
            fullfile = f'{kerases[j]}'
            if fullfile.endswith('.keras'):
                print(f'{len(choices)}: {fullfile}')
                choices.append(fullfile)
    keras_pick = int(input())
    model_filenameFULL = f'{choices[keras_pick]}'
print(f'load: {model_filenameFULL}')
model_full = tf.keras.models.load_model(model_filenameFULL, custom_objects=custom_objects, compile=False)

print('output foldername:')
if outputfoldername:
    print(outputfoldername)
else:
    outputfoldername = input()

hls_folder = f'hls_output/{outputfoldername}'
makedirs(hls_folder, exist_ok=True)

#kopi otomatis file ini agar tercatat konfigurasi model dan quantisasi nya
shutil.copy(os.path.realpath(argv[0]), f'{hls_folder}/{os.path.basename(__file__)}')


print("using cifar10 dataset")
X_test = np.load('npy/c10_X_test_main.npy')
Y_test = np.load('npy/c10_Y_test_main.npy')
batch_size = 256

try:
    true_labels = np.argmax(Y_test, axis=1)
except np.exceptions.AxisError:
    # Jika Y_test adalah 1D array, gunakan langsung
    true_labels = Y_test

print("Trying to strip model_full")
model_full_source = strip_pruning(model_full)
model_full_source.summary()
print("that was model_full_source summary")
# model_target = model_full_source
print("Cloning model to a clean graph...")
with custom_object_scope(custom_objects):
    model_target = tf.keras.models.clone_model(model_full_source)
    model_target.set_weights(model_full_source.get_weights())

# Sanity check tapi pakai forward pass, bukan membangun sub-Model
try:
    dummy = np.zeros((1,) + tuple(model_target.input_shape[1:]), dtype=np.float32)
    _ = model_target.predict(dummy, verbose=0)
    print("Forward pass OK (graph konsisten)")
except Exception as e:
    print(f"Connectivity/forward check failed: {e}")
    # Info diagnostik minimal
    try:
        iq = model_target.get_layer('input_quant')
        print(f"input_quant.input: {iq.input}, model.inputs: {model_target.inputs}")
    except:
        print("Layer 'input_quant' tidak ditemukan.")
    raise

# ----- Konversi ke HLS -----
hls_config = hls4ml.utils.config_from_keras_model(model_target, granularity='name')

# ENABLE TRACE untuk numerical profiling hls4ml
hls_config['Model']['Trace'] = True
for ln in hls_config['LayerName']:
    hls_config['LayerName'][ln]['Trace'] = True

[print(k) for k in hls_config['LayerName'].keys()]
# print("lanjut?")
# input()
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['ReuseFactor'] = 60
hls_config['LayerName']['fused_convbn_0']['ReuseFactor'] = 432
hls_config['LayerName']['fused_convbn_1a']['ReuseFactor'] = 9216
hls_config['LayerName']['fused_convbn_1b']['ReuseFactor'] = 36864
hls_config['LayerName']['fused_convbn_2']['ReuseFactor'] = 73728
hls_config['LayerName']['dense_2']['ReuseFactor'] = 640
hls_config['LayerName']['dense_2']['Precision']['result'] = 'ap_fixed<16,6>'
hls_config['Model']['Precision']['default'] = 'ap_fixed<16,6>'
hls_config['Model']['Precision']['result'] = 'ap_fixed<16,6>'

hls_model = hls4ml.converters.convert_from_keras_model(
    model_target,
    hls_config=hls_config,
    output_dir=hls_folder,
    project_name=outputfoldername,
    part='xc7z020clg400-1',
    backend='Vitis',
    io_type='io_stream',
    interface='none'
)
print(f"akan melakukan hls compile")
hls_model.compile()
print("selesai hls compile")
plot_output = f'img/hls_plot_{outputfoldername}.png'
hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=plot_output)
try:exec2(f'xdg-open', f'{plot_output}')
except:pass
# ----- Cek numerical() -----
if args.skip_profiling:
    print("SKIPPING numerical profiling sesuai parameter --skip-profiling")
else:
    print("Memulai profiling numerik...")
    try:
        from hls4ml.model.profiling import numerical
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Numerical profiling timeout!")
        
        # Set timeout 180 detik (3 menit)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60*10)
        
        # Batasi jumlah sample untuk profiling
        # Hanya n sample pertama
        X_sample = np.ascontiguousarray(X_test[:50]).astype(np.float32)  # pastikan contiguous + float32
        numerical(model=model_target, hls_model=hls_model, X=X_sample)
        
        signal.alarm(0)  # Matikan alarm
        print("Profiling numerik selesai!")
        
    except TimeoutError:
        print("WARNING: Numerical profiling timeout - dilewati untuk melanjutkan")
    except Exception as e:
        print(f"WARNING: Numerical profiling error - {e} - dilewati untuk melanjutkan")

# Buat model HLS lagi untuk test inferensi, tanpa trace

# Bebaskan model trace agar tidak memperlambat
del hls_model

# ----- Konversi untuk INFERENSI (TRACE OFF) -----
hls_config = hls4ml.utils.config_from_keras_model(model_target, granularity='name')
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['ReuseFactor'] = 60
hls_config['LayerName']['fused_convbn_0']['ReuseFactor'] = 432
hls_config['LayerName']['fused_convbn_1a']['ReuseFactor'] = 432 #9216
hls_config['LayerName']['fused_convbn_1b']['ReuseFactor'] = 432 #36864
hls_config['LayerName']['fused_convbn_2']['ReuseFactor'] = 432 #73728
hls_config['LayerName']['dense_2']['ReuseFactor'] = 640
hls_config['LayerName']['dense_2']['Precision']['result'] = 'ap_fixed<16,6>'
hls_config['Model']['Precision']['default'] = 'ap_fixed<16,6>'
hls_config['Model']['Precision']['result'] = 'ap_fixed<16,6>'
# JANGAN set Trace=True

hls_model = hls4ml.converters.convert_from_keras_model(
    model_target,
    hls_config=hls_config,
    output_dir=hls_folder,
    project_name=outputfoldername,
    part='xc7z020clg400-1',
    backend='Vitis',
    io_type='io_stream',
    interface='none'
)
hls_model.compile()
hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=f"{plot_output}.2.png")




print("test inferensi pada model hls")
# ----- Evaluasi akurasi HLS model -----
chunk = 500
n = X_test.shape[0]
print(f"Predict in chunks of {chunk} (total {n})")

# Single HLS model
logits_chunks = []
chunk_times = []
start_total = time.perf_counter()
for start in range(0, n, chunk):
    end = min(start + chunk, n)
    print(f"{end:5d}", end="")
    t0 = time.perf_counter()
    x_ch = np.ascontiguousarray(X_test[start:end]).astype(np.float32)
    log_ch = hls_model.predict(x_ch)
    dt = time.perf_counter() - t0
    chunk_times.append(dt)
    print(f"  {dt*1000:.1f} ms")
    logits_chunks.append(log_ch)
logits = np.concatenate(logits_chunks, axis=0)

total_time = time.perf_counter() - start_total
avg_time = (sum(chunk_times) / len(chunk_times)) if chunk_times else 0.0
ms_per_img = (total_time / max(n, 1)) * 1000.0
print(f"Total inferensi: {total_time:.3f} s | Rata2/chunk: {avg_time:.3f} s | Throughput: {n/total_time:.1f} img/s | {ms_per_img:.3f} ms/img")

probs_hls = tf.nn.softmax(logits).numpy()
try:
    pred_labels_hls = np.argmax(probs_hls, axis=1)
except np.exceptions.AxisError:
    # Jika probs_hls adalah 1D array, gunakan langsung
    pred_labels_hls = probs_hls
hls_acc = accuracy_score(true_labels, pred_labels_hls)
np.save('npy/y_hlsmodel.npy',pred_labels_hls)
print(f"Akurasi simulasi HLS: {hls_acc * 100:.2f}%")
with open(f"hls_output/{outputfoldername}/acc_{hls_acc * 100:.2f}.txt", 'w') as f:
    f.write(f"Akurasi simulasi HLS: {hls_acc * 100:.2f}%\n")
    f.write(f"Total inferensi: {total_time:.3f} s | Rata2/chunk: {avg_time:.3f} s | Throughput: {n/total_time:.1f} img/s | {ms_per_img:.3f} ms/img\n")
print("lanjut..")
if args.minim:
    print("Membuat Vitis project minimal")
    hls_model.build(csim=False, synth=False, vsynth=False, export=False, cosim=False)
else:
    hls_model.build(csim=True, synth=True, vsynth=True, export=True, cosim=True)


    print(exec(f"cat hls_output/{outputfoldername}/vivado_synth.rpt  | grep '4. IO and' -A 7"))
    print(exec(f"cat hls_output/{outputfoldername}/vivado_synth.rpt  | grep 'Slice Logic' -A 16"))
    print(exec(f"cat hls_output/{outputfoldername}/{outputfoldername}_prj/solution1/syn/report/{outputfoldername}_csynth.rpt  | grep 'Utilization Estim' -A 20"))
