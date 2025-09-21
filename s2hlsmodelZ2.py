class Abstract(object):pass

from os import uname
from os import listdir
from os import makedirs
from os.path import realpath, basename
os = Abstract();os.path=Abstract();
os.path.realpath = realpath
os.path.basename = basename

import shutil
import time
from sys import argv
import argparse

import tensorflow as tf
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.keras.models import Model
from qkeras import QActivation, QDense, QConv2DBatchnorm
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


# layer custom (dari QKeras, wrapper pruning)
custom_objects = {
    'QActivation': QActivation,
    'QDense': QDense,
    'QConv2DBatchnorm': QConv2DBatchnorm,
    'PruneLowMagnitude': PruneLowMagnitude
}


parser = argparse.ArgumentParser()
parser.add_argument('--sambung', action='store_true')
parser.add_argument('--input', type=str, required=False, default='')
parser.add_argument('--output', type=str, required=False, default='')
parser.add_argument('--minim', action='store_true', help='bikin vitis project minimal')
args = parser.parse_args()

input_modelfull = args.input
outputfoldername = args.output
sambung = args.sambung

print(f'quick usage untuk 1 model full:')
print(f'`python {os.path.basename(__file__)} --sambung --input=[filename model full] --output=[output folder name]`')

if input_modelfull and outputfoldername:
    sambung = True

# ----- Load model and test data -----
kerases_folder = listdir('keras/')
kerases_folder.sort()
kerases_folder = [f'keras/{f}' for f in kerases_folder]
print("modelA akan disambung ke modelB")
choices = []
for i in range(len(kerases_folder)):
    print(f'{kerases_folder[i]}:')
    kerases = listdir(kerases_folder[i])
    kerases = [f'{kerases_folder[i]}/{f}' for f in kerases]
    for j in range(len(kerases)):
        fullfile = f'{kerases[j]}'
        print(f'{len(choices)}: {fullfile}')
        choices.append(fullfile)
if outputfoldername:
    sambung = False
else:
    print("pakai 2 model yang di-sambung? y/n\n(n=pakai 1 model full)")
    sambung = (input()=='y')

if (sambung):
    print('pilih modelA: [1/2/3..]')
    keras_pick = int(input())
    model_filenameA = f'{choices[keras_pick]}'
    print('pilih modelB: [1/2/3..]')
    keras_pick = int(input())
    model_filenameB = f'{choices[keras_pick]}'

    print(f'load: {model_filenameA}')
    model_A_conv = tf.keras.models.load_model(model_filenameA, custom_objects=custom_objects)
    model_B_dense = tf.keras.models.load_model(model_filenameB, custom_objects=custom_objects)
else:
    print('pilih model FULL: [1/2/3..]')
    if input_modelfull:
        model_filenameFULL = input_modelfull
    else:
        keras_pick = int(input())
        model_filenameFULL = f'{choices[keras_pick]}'
    print(f'load: {model_filenameFULL}')
    model_full = tf.keras.models.load_model(model_filenameFULL, custom_objects=custom_objects)

print('output foldername:')
if outputfoldername:
    print(outputfoldername)
else:
    outputfoldername = input()

hls_folder = f'hls_output/{outputfoldername}'
makedirs(hls_folder, exist_ok=True)

#kopi otomatis file ini agar tercatat konfigurasi model dan quantisasi nya
shutil.copy(os.path.realpath(argv[0]), f'{hls_folder}/{os.path.basename(__file__)}')



X_test = np.load('npy/X_test_main.npy')
Y_test = np.load('npy/Y_test_main.npy')
batch_size = 128
n_classes = 10

try:
    true_labels = np.argmax(Y_test, axis=1)
except np.exceptions.AxisError:
    # Jika Y_test adalah 1D array, gunakan langsung
    true_labels = Y_test

if sambung:
    model_A_conv_pruned = strip_pruning(model_A_conv)
    model_B_dense_pruned = strip_pruning(model_B_dense)
    model_target = model_A_conv_pruned
else:
    print("Trying to strip model_full")
    model_full_source = strip_pruning(model_full)
    model_full_source.summary()
    print("that was model_full_source summary")
    # model_target = model_full_source

    # Temukan layer QDense dan BatchNormalization yang berurutan
    print("Melakukan folding BatchNormalization ke Dense layer...")
    dense_to_fold = model_full_source.get_layer('dense_1')
    bn_to_fold = model_full_source.get_layer('bn_3')

    # fold layer dense+bn
    folded_dense = fold_dense_bn(dense_to_fold, bn_to_fold)

    # membangun model baru dari awal, layer per layer,
    # hanya menyalin konfigurasi dan bobot. Ini menjamin grafik yang bersih.
    print("Membangun ulang model secara penuh untuk memastikan grafik yang bersih...")

    def get_new_layer(old_layer, input_shape=None):
        """Helper untuk membuat layer baru dari layer lama."""
        config = old_layer.get_config()
        new_layer = old_layer.__class__.from_config(config)
        # Build layer jika perlu sebelum set_weights
        if old_layer.get_weights():
            shape = input_shape
            if shape is None:
                try:
                    shape = old_layer.input_shape
                except AttributeError:
                    shape = None
            if shape is not None:
                # Buat dummy input untuk memanggil layer (agar weights terinisialisasi)
                # shape: (batch, ...) -> batch=1
                dummy_shape = [1] + list(shape[1:])
                dummy_input = tf.zeros(dummy_shape)
                try:
                    new_layer(dummy_input)
                except Exception as e:
                    print(f"Warning: gagal memanggil {old_layer.name} dengan dummy input: {e}")
        if old_layer.get_weights():
            new_layer.set_weights(old_layer.get_weights())
        return new_layer

    # 1. Buat input baru
    new_input = tf.keras.layers.Input(shape=model_full_source.input_shape[1:], name="input_1")
    x = new_input

    # 2. Bangun kembali semua layer SEBELUM bagian yang di-fold
    layers_to_rebuild_head = [
        'input_quant', 'fused_convbn_0', 'conv_act_0', 'max_pooling2d',
        'fused_convbn_1a', 'conv_act_1a', 'max_pooling2d_1',
        'fused_convbn_1b', 'conv_act_1b', 'max_pooling2d_2',
        'fused_convbn_2',
        'conv_act_2', 'conv_maxpool', 'conv_globalavg'
    ]
    for layer_name in layers_to_rebuild_head:
        old_layer = model_full_source.get_layer(layer_name)
        new_layer = get_new_layer(old_layer, input_shape=x.shape)
        x = new_layer(x)

    # 3. Masukkan layer yang sudah di-fold
    # `folded_dense` sudah merupakan layer baru dengan bobot yang benar
    x = folded_dense(x)

    # 4. Bangun kembali semua layer SETELAH bagian yang di-fold
    layers_to_rebuild_tail = [
        'dense_act_0', 'dense_2'
    ]
    for layer_name in layers_to_rebuild_tail:
        new_layer = get_new_layer(model_full_source.get_layer(layer_name))
        x = new_layer(x)

    # 5. Output akhir adalah tensor 'x' dari layer terakhir
    outputs = x

    # 6. Buat model final dari input dan output yang baru
    model_target = Model(inputs=new_input, outputs=outputs, name="model_folded")
    print("Model baru 'model_folded' telah dibuat.")
    model_target.summary()
    print('strip pruning')
    model_target = strip_pruning(model_target)
    

# ----- Konversi ke HLS -----
hls_config = hls4ml.utils.config_from_keras_model(model_target, granularity='name')

[print(k) for k in hls_config['LayerName'].keys()]
# print("lanjut?")
# input()
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['ReuseFactor'] = 60
# hls_config['Model']['Precision'] = 'ap_fixed<16,6>' #bilangan kedua untuk integer bit termasuk sign bit

# hls_config['LayerName']['fused_convbn_1']['Precision']['weight'] = 'ap_fixed<12,1>'
# hls_config['LayerName']['fused_convbn_1']['Precision']['bias'] = 'ap_fixed<12,1>'
# hls_config['LayerName']['fused_convbn_1']['Precision']['result'] = 'ap_fixed<8,1>'  # output 
# hls_config['LayerName']['fused_convbn_2']['Precision']['weight'] = 'ap_fixed<12,1>'
# hls_config['LayerName']['fused_convbn_2']['Precision']['bias'] = 'ap_fixed<12,1>'
# hls_config['LayerName']['fused_convbn_2']['Precision']['result'] = 'ap_fixed<8,1>'  # output 
hls_config['LayerName']['fused_convbn_2']['implementation'] = 'dsp'  # 
# hls_config['LayerName']['fused_convbn_2']['directive'] = 'BIND_OP op=mul impl=dsp'  # #pragma HLS BIND_OP op=mul impl=dsp, untuk layerke2 dicoba, bisakah pakai dsp

# for layer_name in hls_config['LayerName']:
#     if 'dense' in layer_name:
#         hls_config['LayerName'][layer_name]['Precision'] = 'ap_fixed<16,8>'

        # hls_config['LayerName'][layer_name]['Precision']['weight'] = 'ap_fixed<16,7>'
        # hls_config['LayerName'][layer_name]['Precision']['bias'] = 'ap_fixed<16,7>'
        # hls_config['LayerName'][layer_name]['Precision']['result'] = 'ap_fixed<8,2>'  # output 
hls_config['LayerName']['dense_2']['Precision']['result'] = 'ap_fixed<8,4>'  # output 

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
plot_output = f'img/hls_plot_{outputfoldername}.png'
hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=plot_output)
try:exec2(f'xdg-open', f'{plot_output}')
except:pass
# ----- Cek numerical() -----
from hls4ml.model.profiling import numerical
numerical(model=model_target, hls_model=hls_model)

# ----- Evaluasi akurasi HLS model -----
# logits_hls = hls_model.predict(np.ascontiguousarray(X_test))
if sambung:
    # Inference: FPGA (PL) part
    conv_out = hls_model.predict(np.ascontiguousarray(X_test))
    # Inference: CPU (PS) part
    logits = model_B_dense_pruned.predict(conv_out)
else:
    logits = hls_model.predict(np.ascontiguousarray(X_test))

probs_hls = tf.nn.softmax(logits).numpy()
try:
    pred_labels_hls = np.argmax(probs_hls, axis=1)
except np.exceptions.AxisError:
    # Jika probs_hls adalah 1D array, gunakan langsung
    pred_labels_hls = probs_hls
hls_acc = accuracy_score(true_labels, pred_labels_hls)
np.save('npy/y_hlsmodel.npy',pred_labels_hls)
print(f"Akurasi simulasi HLS: {hls_acc * 100:.2f}%")
print("lanjut dalam 5 detik..")
l = 'y'
time.sleep(5)
if args.minim:
    print("Membuat Vitis project minimal")
    hls_model.build(csim=False, synth=False, vsynth=False, export=True, cosim=False)
else:
    hls_model.build(csim=True, synth=True, vsynth=True, export=True, cosim=True)


    print(exec(f"cat hls_output/{outputfoldername}/vivado_synth.rpt  | grep '4. IO and' -A 7"))
    print(exec(f"cat hls_output/{outputfoldername}/vivado_synth.rpt  | grep 'Slice Logic' -A 16"))
    print(exec(f"cat hls_output/{outputfoldername}/{outputfoldername}_prj/solution1/syn/report/{outputfoldername}_csynth.rpt  | grep 'Utilization Estim' -A 20"))
