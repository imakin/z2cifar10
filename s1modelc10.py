from os import uname, listdir, makedirs
import shutil
import argparse

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, BatchNormalization, Dropout, ZeroPadding2D, RandomCrop, RandomFlip
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm

from s0dataset import datasets

from sys import argv

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=False, default=None)
filterdefaults = [32,64,128,64,10]
prunedefaults = [0.1, 0.1, 0.1, 0.2, 0.1]
for x in enumerate(filterdefaults):
    i = x[0]
    banyak_filter = x[1]
    parser.add_argument(f'--filter{i}', type=int, required=False, default=banyak_filter)
    parser.add_argument(f'--prune{i}', type=float, required=False, default=prunedefaults[i])

args = parser.parse_args()
for k, v in vars(args).items():
    print(f"{k}: {v}")

print("Nama Folder file keras: ")
suffix = args.name
if not suffix:
    suffix = input()
print(suffix)
model_conv_name = f'keras/{suffix}/main_conv.keras'
model_dense_name = f'keras/{suffix}/main_dense.keras'
model_full_name = f'keras/{suffix}/main_full.keras'

makedirs(f'keras/{suffix}', exist_ok=True)

print(f'Akan menyimpan pada {model_full_name}')

#kopi otomatis file ini agar tercatat konfigurasi model dan quantisasi nya
shutil.copy(argv[0], f'keras/{suffix}/{argv[0]}')


# --- Dataset ---
BATCH_SIZE = 256
ds = datasets.c10
train_data = ds.train
val_data = ds.val
test_data = ds.test



def pruneVarious(layer, sparsity_target={}):
    try:
        starget = sparsity_target[layer.name]
    except:
        return layer # dont prune
    
    NSTEPS = int(np.ceil(len(train_data) if hasattr(train_data, '__len__') else 1))
    pruning_params = {
        'pruning_schedule': PolynomialDecay(
            initial_sparsity=0.0,      # Mulai dari 0% bobot nol
            final_sparsity=starget,       # Target akhir berapa % bobot jadi nol
            begin_step=NSTEPS * 2,     # Mulai pruning setelah 2 epoch
            end_step=NSTEPS * 40,      # Selesai pruning pada epoch ke-40
            frequency=NSTEPS           # Update mask pruning setiap 1 epoch
        )
    }
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

def make_sparse_cce_with_label_smoothing(smoothing=0.05, num_classes=10):
    """Create a loss that applies label smoothing for sparse integer labels.
    Works on older TF that lack label_smoothing in SparseCategoricalCrossentropy.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    def loss(y_true, y_pred):
        # Ensure shape is (batch,) not (batch,1) before one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        smooth = tf.cast(smoothing, y_pred.dtype)
        num_c = tf.cast(num_classes, y_pred.dtype)
        y_true_smoothed = (1.0 - smooth) * y_true_oh + smooth / num_c
        return cce(y_true_smoothed, y_pred)
    return loss





# --- Model Split ---
shapes = None
for x,y in val_data.take(1):
    shapes = x.shape
input_shape = shapes[1:]  # ambil shape tanpa batch size. (32,32,3)

# FPGA model: 3 Conv2D + batchnormalization + MaxPooling + Flatten + quantized
inputs = Input(shape=input_shape)

# # Augmentasi hanya aktif saat training; inference akan melewati transformasi acak
aug = tf.keras.Sequential([
    ZeroPadding2D(4),
    RandomCrop(32, 32),
    RandomFlip('horizontal')
], name='aug')
x = aug(inputs)
# QActivation mengkuantisasi input [0,1]; QKeras versi ini tidak mendukung keep_sign arg.
# Pakai bits=10, integer=0 (signed default) → efektif Q0.9, aman untuk nilai non-negatif.
x = QActivation('quantized_bits(bits=16,integer=0,alpha=1)', name='input_quant')(x)
x = QConv2DBatchnorm(
    args.filter0,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same', # 'same' bikin layer zero padding
    kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)", #more fractional bits
    bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)", #more integer bits, use_bias
    kernel_initializer='lecun_uniform',
    kernel_regularizer=l2(1e-4),
    use_bias=True,
    name='fused_convbn_0'
)(x)
x = QActivation('quantized_relu(bits=10,integer=2)', name='conv_act_0')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)", #bilangan kedua (0) integer bit tidak termasuk sign bit
    bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    kernel_regularizer=l2(1e-4),
    use_bias=True,
    name='fused_convbn_1a'
)(x)
x = QActivation('quantized_relu(bits=10,integer=2)', name='conv_act_1a')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)", #bilangan kedua (0) integer bit tidak termasuk sign bit
    bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    kernel_regularizer=l2(1e-4),
    use_bias=True,
    name='fused_convbn_1b'
)(x)
x = QActivation('quantized_relu(bits=10,integer=2)', name='conv_act_1b')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter2,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
    bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    kernel_regularizer=l2(1e-4),
    use_bias=True,
    name='fused_convbn_2'
)(x) # fused_convbn_2 menghasilkan output dengan shape (batch, h, w, filters)
x = QActivation('quantized_relu(bits=12,integer=3)', name='conv_act_2')(x)
# x = MaxPooling2D(name='conv_maxpool')(x) #tanpa maxpooling cifar10
# nilai/channel. MaxPool sebelum GAP tidak diperlukan.
# Hilang informasi: Dengan 3 pool, fitur jadi 4x4. Pool lagi ke 2x2 lalu GAP berarti merata-rata 4 elemen, bukan 16 elemen. Statistik jadi lebih bising dan sinyal gradien lebih lemah.
# Robust terhadap kuantisasi/pruning: GAP pada 4x4 memberi averaging lebih besar → lebih stabil terhadap noise quant/pruning dibanding 2x2.
# Biaya hampir nol: Tidak ada layer konvolusi setelahnya, jadi menghapus MaxPool terakhir tidak menambah MAC konvolusi. Hanya menambah sedikit elemen yang dirata-rata oleh GAP.

# GlobalAveragePooling2D merata-ratakan setiap channel (filter) menjadi satu nilai, sehingga output-nya menjadi (batch, filters).
x = tf.keras.layers.GlobalAveragePooling2D(name='conv_globalavg')(x)

y = QDense(
    args.filter3,
    kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
    kernel_regularizer=l2(1e-4),
    use_bias=False,
    name='dense_1'
)(x)
y = BatchNormalization(
    name='bn_3'
)(y)
y = QActivation(
    'quantized_relu(12,3)',
    name='dense_act_0'
)(y)
y = Dropout(0.3, name='dense_dropout_0')(y)
outputs = QDense(
    args.filter4, #default 10, cifar10 ada 10 kelas
    kernel_quantizer="quantized_bits(bits=8,integer=3,alpha=1)",
    kernel_regularizer=l2(1e-4),
    name='dense_2'
)(y)
model_full_unpruned = Model(inputs, outputs, name="model_full")




def pruneF(layer):
    return pruneVarious(layer, sparsity_target={
        'fused_convbn_0': args.prune0, #default 0.1,
        'fused_convbn_1a': args.prune1, #default 0.1,
        'fused_convbn_1b': args.prune2, # 0.1,
        'fused_convbn_2': args.prune3, # 0.2,
        'dense_1': args.prune4, # 0.1,
    })

# Prune semua model split sebelum training
full_model = tf.keras.models.clone_model(model_full_unpruned, clone_function=pruneF)


# compile

# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
for target in ([
    full_model
]):
    target.compile(
        optimizer=tf.keras.optimizers.Adam(0.002),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #bila one-hot enc
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #label kita 0 dan 1
        # loss=make_sparse_cce_with_label_smoothing(0.05, num_classes=args.filter4),
        metrics=['accuracy']
    )

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=4, min_lr=1e-6, verbose=1),
    pruning_callbacks.UpdatePruningStep(),  # Uncomment jika menggunakan pruning
]
"""
Fit bebarengan:
    full_model menggabungkan model_conv dan model_dense secara end-to-end.
    Saat .fit() dipanggil pada full_model, seluruh parameter
    (baik di model_conv maupun model_dense) akan di-train bersama.
    model_conv.fit atau model_dense.fit hanya digunakan jika ingin melatih
    bagian itu saja (biasanya tidak dilakukan pada skenario split training end-to-end).
"""
full_model.fit(
    train_data,
    epochs=50,
    verbose=2,
    validation_data=val_data,
    callbacks=callbacks
)




def build_inference_no_aug_softcoded(trained_model):
    # Tentukan layer apa saja yang ingin Anda LEWATI
    # Gunakan set ({}) untuk pencarian yang lebih cepat
    layers_to_skip = {
        'aug',                # Layer augmentasi
        'dense_dropout_0',    # Layer dropout
        'input_1'             # Layer input asli (karena kita akan buat yang baru)
    }

    # Mulai membangun graf baru
    inp = tf.keras.Input(shape=trained_model.input_shape[1:], name='input_1')
    x = inp

    # Iterasi semua layer di model asli secara dinamis
    for layer in trained_model.layers:
        # Jika nama layer ada di daftar skip, abaikan layer tersebut
        if layer.name in layers_to_skip:
            continue
            
        # Jika bukan layer yang di-skip, sambungkan ke graf baru
        x = layer(x)

    # Buat model baru dari input baru dan output graf yang sudah dimodifikasi
    return tf.keras.Model(inp, x, name='model_full_no_aug_softcoded')

# --- Cara Pakai ---
# Asumsikan 'full_model' sudah ada dan terdefinisi

full_model = build_inference_no_aug_softcoded(full_model)

# --- Testing ---
X_test, Y_test = [], []
for x, y in test_data.unbatch().take(10000):  # ambil sebagian jika dataset besar
    X_test.append(x.numpy())
    Y_test.append(y.numpy())
X_test = np.stack(X_test)
Y_test = np.stack(Y_test)

dense_out_logits = full_model.predict(X_test, batch_size=BATCH_SIZE)
probs = tf.nn.softmax(dense_out_logits).numpy()
pred_labels = np.argmax(probs, axis=1)
# true_labels = np.argmax(Y_test, axis=1) # bila one hot encoding
true_labels = Y_test  # langsung saja, tidak perlu argmax, krn sudah integer
acc = accuracy_score(true_labels, pred_labels)
print(f"Akurasi model : {acc * 100:.2f}%")



np.save('npy/c10_X_test_main.npy', X_test)
np.save('npy/c10_Y_test_main.npy', Y_test)
np.save('npy/c10_dense_out_logits_main.npy', dense_out_logits)

np.save(f'keras/{suffix}/X_test_main.npy', X_test)
np.save(f'keras/{suffix}/Y_test_main.npy', Y_test)
np.save(f'keras/{suffix}/dense_out_logits_main.npy', dense_out_logits)


print("data test tersimpan")

full_model.save(model_full_name)
print(f"Model gabungan conv+dense tersimpan di {model_full_name}")

with open(f'keras/{suffix}/model_full_summary.txt', 'w') as f:
    full_model.summary(print_fn=lambda x: f.write(x + '\n'))
akurasi_text = f"acc_{acc * 100:.2f}"
with open(f'keras/{suffix}/{akurasi_text}', 'w') as f:
    f.write(f"isian filters: {args.filter0}, {args.filter1}, {args.filter2}, {args.filter3}, {args.filter4}\n")

print("\n\nls keras/")
[print(f) for f in listdir('keras')]
for k, v in vars(args).items():
    print(f"{k}: {v}")