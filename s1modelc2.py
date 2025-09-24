from os import uname, listdir, makedirs
import shutil
import argparse

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, BatchNormalization
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
filterdefaults = [16,12,8,8,2]
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
BATCH_SIZE = 1024
ds = datasets.c2
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
            end_step=NSTEPS * 10,      # Selesai pruning pada epoch ke-10
            frequency=NSTEPS           # Update mask pruning setiap 1 epoch
        )
    }
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)





# --- Model Split ---
shapes = None
for x,y in val_data.take(1):
    shapes = x.shape
input_shape = shapes[1:]  # ambil shape tanpa batch size. (32,32,3)

# FPGA model: 3 Conv2D + batchnormalization + MaxPooling + Flatten + quantized
inputs = Input(shape=input_shape)

#QActivation ini ngubah tipe data pada input dari float32 ke quantisasi yang dipilih, misal 8bit <8,1> atau (16,4,alpha=1) (8bit, 0integer, 1 sign, 7frac)
x = QActivation('quantized_bits(bits=16,integer=5,alpha=1)', name='input_quant')(inputs)
x = QConv2DBatchnorm(
    args.filter0, #default 10,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same', # 'same' bikin layer zero padding
    kernel_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
    bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    use_bias=True,
    name='fused_convbn_0'
)(x)
x = QActivation('quantized_relu(bits=9,integer=4)', name='conv_act_0')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter1, #default 8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=9,integer=4,alpha=1)", #bilangan kedua (0) integer bit tidak termasuk sign bit
    bias_quantizer="quantized_bits(bits=9,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    use_bias=True,
    name='fused_convbn_1a'
)(x)
x = QActivation('quantized_relu(bits=9,integer=4)', name='conv_act_1a')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter1, #default 8,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=9,integer=4,alpha=1)", #bilangan kedua (0) integer bit tidak termasuk sign bit
    bias_quantizer="quantized_bits(bits=9,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    use_bias=True,
    name='fused_convbn_1b'
)(x)
x = QActivation('quantized_relu(bits=9,integer=4)', name='conv_act_1b')(x)
x = MaxPooling2D()(x)

x = QConv2DBatchnorm(
    args.filter2, #default 3,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    kernel_quantizer="quantized_bits(bits=9,integer=4,alpha=1)",
    bias_quantizer="quantized_bits(bits=9,integer=4,alpha=1)",
    kernel_initializer='lecun_uniform',
    use_bias=True,
    name='fused_convbn_2'
)(x) # fused_convbn_2 menghasilkan output dengan shape (batch, h, w, filters)
x = QActivation('quantized_relu(bits=9,integer=4)', name='conv_act_2')(x)
x = MaxPooling2D(name='conv_maxpool')(x)
# GlobalAveragePooling2D merata-ratakan setiap channel (filter) menjadi satu nilai, sehingga output-nya menjadi (batch, filters).
x = tf.keras.layers.GlobalAveragePooling2D(name='conv_globalavg')(x)

y = QDense(
    args.filter3, #default 8,
    kernel_quantizer="quantized_bits(bits=9,integer=4,alpha=1)",
    use_bias=False,
    name='dense_1'
)(x)
y = BatchNormalization(
    name='bn_3'
)(y)
y = QActivation(
    'quantized_relu(9,2)',
    name='dense_act_0'
)(y)
outputs = QDense(
    args.filter4, #default 3,
    kernel_quantizer="quantized_bits(bits=8,integer=4,alpha=1)",
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
        optimizer=tf.keras.optimizers.Adam(0.001),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #bila one-hot enc
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #label kita 0 dan 1
        metrics=['accuracy']
    )

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=4, min_lr=1e-6, verbose=1),
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



np.save('npy/c2_X_test_main.npy', X_test)
np.save('npy/c2_Y_test_main.npy', Y_test)
np.save('npy/c2_dense_out_logits_main.npy', dense_out_logits)

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