"""
Filter Pruning - Menghapus entire filters yang kurang penting
Berbeda dengan weight pruning yang hanya set weights=0

Cara kerja:
1. Train model normal dulu
2. Hitung importance setiap filter (berdasarkan L1-norm atau magnitude)
3. Hapus filter dengan importance terendah
4. Fine-tune model yang sudah dipangkas
"""

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

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm

try:
    from s0dataset import datasets
except ImportError:
    from s0dataseth import datasets

from sys import argv

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=False, default=None)
parser.add_argument('--pretrained', type=str, required=False, default=None, help='Path to pretrained model to prune')

# Filter counts - akan di-reduce secara otomatis berdasarkan prune ratio
filterdefaults = [32,64,128,64,10]
# Prune ratio: berapa persen filter yang akan DIHAPUS (0.3 = hapus 30% filter)
prunedefaults = [0.0, 0.0, 0.0, 0.0, 0.0]

for x in enumerate(filterdefaults):
    i = x[0]
    banyak_filter = x[1]
    parser.add_argument(f'--filter{i}', type=int, required=False, default=banyak_filter)
    parser.add_argument(f'--prune{i}', type=float, required=False, default=prunedefaults[i])

parser.add_argument(f'--filter1a', type=int, required=False, default=0)
parser.add_argument(f'--filter1b', type=int, required=False, default=0)
parser.add_argument(f'--filter1c', type=int, required=False, default=0)

parser.add_argument(f'--prune1a', type=float, required=False, default=0.0)
parser.add_argument(f'--prune1b', type=float, required=False, default=0.0)
parser.add_argument(f'--prune1c', type=float, required=False, default=0.0)

args = parser.parse_args()
if args.filter1a==0:
    args.filter1a = args.filter1
if args.filter1b==0:
    args.filter1b = args.filter1
if args.prune1a==0.0:
    args.prune1a = args.prune1
if args.prune1b==0.0:
    args.prune1b = args.prune1

for k, v in vars(args).items():
    print(f"{k}: {v}")

print("Nama Folder file keras: ")
suffix = args.name
if not suffix:
    suffix = input()
print(suffix)
model_name = f'keras/{suffix}/main_full.keras'

makedirs(f'keras/{suffix}', exist_ok=True)
print(f'Akan menyimpan pada {model_name}')

#kopi otomatis file ini agar tercatat konfigurasi model dan quantisasi nya
shutil.copy(argv[0], f'keras/{suffix}/{argv[0]}')


# --- Dataset ---
BATCH_SIZE = 256
ds = datasets.c10
train_data = ds.train
val_data = ds.val
test_data = ds.test


def calculate_filter_importance(layer):
    """
    Hitung importance setiap filter berdasarkan L1-norm
    Filter dengan norm kecil = kurang penting = kandidat untuk dihapus
    """
    weights = layer.get_weights()[0]  # [h, w, in_ch, out_ch] untuk Conv2D
    
    if len(weights.shape) == 4:  # Conv2D
        # Hitung L1-norm untuk setiap output filter
        # Sum across height, width, dan input channels
        importance = np.sum(np.abs(weights), axis=(0, 1, 2))
    elif len(weights.shape) == 2:  # Dense
        # Sum across input dimension
        importance = np.sum(np.abs(weights), axis=0)
    else:
        return None
    
    return importance


def prune_filters_from_layer(layer, prune_ratio, layer_name_new):
    """
    Prune filters dari layer berdasarkan importance
    Returns: layer baru dengan filter yang lebih sedikit
    """
    weights = layer.get_weights()
    kernel = weights[0]
    bias = weights[1] if len(weights) > 1 else None
    
    # Hitung importance
    importance = calculate_filter_importance(layer)
    if importance is None:
        print(f"  ⚠️  Layer {layer.name} tidak bisa di-prune (shape tidak didukung)")
        return layer
    
    # Tentukan berapa filter yang mau dipertahankan
    n_filters = len(importance)
    n_keep = int(n_filters * (1 - prune_ratio))
    n_keep = max(1, n_keep)  # minimal 1 filter
    
    # Ranking berdasarkan importance (descending)
    keep_indices = np.argsort(importance)[-n_keep:]
    keep_indices = np.sort(keep_indices)  # sort untuk maintain order
    
    print(f"  Filter pruning {layer.name}: {n_filters} → {n_keep} filters ({prune_ratio*100:.0f}% pruned)")
    print(f"    Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    
    # Extract weights untuk filter yang dipertahankan
    if len(kernel.shape) == 4:  # Conv2D
        new_kernel = kernel[:, :, :, keep_indices]
        new_bias = bias[keep_indices] if bias is not None else None
        
        # Buat layer baru dengan jumlah filter yang lebih sedikit
        config = layer.get_config()
        config['filters'] = n_keep
        config['name'] = layer_name_new
        
        if isinstance(layer, QConv2DBatchnorm):
            new_layer = QConv2DBatchnorm.from_config(config)
        else:
            new_layer = Conv2D.from_config(config)
    
    elif len(kernel.shape) == 2:  # Dense
        new_kernel = kernel[:, keep_indices]
        new_bias = bias[keep_indices] if bias is not None else None
        
        config = layer.get_config()
        config['units'] = n_keep
        config['name'] = layer_name_new
        
        if isinstance(layer, QDense):
            new_layer = QDense.from_config(config)
        else:
            new_layer = Dense.from_config(config)
    
    # Set weights baru
    new_layer.build(layer.input_shape)
    if new_bias is not None:
        new_layer.set_weights([new_kernel, new_bias])
    else:
        new_layer.set_weights([new_kernel])
    
    return new_layer, keep_indices


def prune_model_filters(model, prune_config):
    """
    Prune filters dari model dan rebuild dengan connectivity yang benar
    
    prune_config: dict dengan layer_name: prune_ratio
    Contoh: {'fused_convbn_0': 0.3, 'fused_convbn_1a': 0.4}
    """
    print("\n" + "="*60)
    print("FILTER PRUNING - Menghapus entire filters")
    print("="*60)
    
    # Track indices yang di-keep untuk adjust next layer
    kept_indices = {}
    
    # Rebuild model layer by layer
    inp = model.input
    x = inp
    
    for layer in model.layers[1:]:  # skip input layer
        layer_name = layer.name
        
        # Skip non-trainable layers
        if not layer.weights:
            x = layer(x)
            continue
        
        # Check jika layer ini perlu di-prune
        if layer_name in prune_config and prune_config[layer_name] > 0:
            # Prune filters
            new_layer, keep_idx = prune_filters_from_layer(
                layer, 
                prune_config[layer_name],
                layer_name + '_pruned'
            )
            kept_indices[layer_name] = keep_idx
            x = new_layer(x)
        else:
            # Tidak di-prune, tapi mungkin perlu adjust input channels
            prev_layer_name = None
            # Cari layer sebelumnya yang di-prune
            for prev_name in kept_indices.keys():
                if prev_name in layer_name or True:  # simplifikasi
                    prev_layer_name = prev_name
            
            if prev_layer_name and prev_layer_name in kept_indices:
                # Adjust input channels
                weights = layer.get_weights()
                kernel = weights[0]
                
                if len(kernel.shape) == 4:  # Conv2D
                    keep_idx = kept_indices[prev_layer_name]
                    new_kernel = kernel[:, :, keep_idx, :]
                    new_weights = [new_kernel] + weights[1:]
                    
                    config = layer.get_config()
                    if isinstance(layer, QConv2DBatchnorm):
                        new_layer = QConv2DBatchnorm.from_config(config)
                    else:
                        new_layer = Conv2D.from_config(config)
                    
                    new_layer.build((None, x.shape[1], x.shape[2], len(keep_idx)))
                    new_layer.set_weights(new_weights)
                    x = new_layer(x)
                else:
                    x = layer(x)
            else:
                x = layer(x)
    
    new_model = Model(inputs=inp, outputs=x, name=model.name + '_pruned')
    return new_model


# ============= MAIN TRAINING FLOW =============

if args.pretrained:
    print(f"Loading pretrained model: {args.pretrained}")
    from tensorflow.keras.utils import custom_object_scope
    custom_objects = {
        'QActivation': QActivation,
        'QDense': QDense,
        'QConv2DBatchnorm': QConv2DBatchnorm,
    }
    with custom_object_scope(custom_objects):
        model_full = tf.keras.models.load_model(args.pretrained, compile=False)
    
    # Prune filters
    prune_config = {
        'fused_convbn_0': args.prune0,
        'fused_convbn_1a': args.prune1a,
        'fused_convbn_1b': args.prune1b,
        'fused_convbn_2': args.prune2,
    }
    
    # Hapus entry dengan prune_ratio = 0
    prune_config = {k: v for k, v in prune_config.items() if v > 0}
    
    if prune_config:
        model_full = prune_model_filters(model_full, prune_config)
        
        # Fine-tune setelah pruning
        print("\nFine-tuning after filter pruning...")
        model_full.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),  # LR lebih kecil
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ]
        
        model_full.fit(
            train_data,
            epochs=30,  # fine-tuning lebih cepat
            verbose=2,
            validation_data=val_data,
            callbacks=callbacks
        )
    else:
        print("No pruning specified (all prune ratios = 0)")
else:
    # Train from scratch dengan jumlah filter yang sudah dikurangi
    print("Training from scratch with reduced filter counts")
    
    # Calculate actual filter counts setelah pruning
    actual_filters = {
        0: int(args.filter0 * (1 - args.prune0)),
        '1a': int(args.filter1a * (1 - args.prune1a)),
        '1b': int(args.filter1b * (1 - args.prune1b)),
        '1c': int(args.filter1c * (1 - args.prune1c)) if args.filter1c > 0 else 0,
        2: int(args.filter2 * (1 - args.prune2)),
        3: int(args.filter3 * (1 - args.prune3)),
        4: args.filter4,  # output layer tidak di-prune
    }
    
    print("Actual filter counts after pruning:")
    for k, v in actual_filters.items():
        print(f"  Layer {k}: {v} filters")
    
    # Build model dengan filter counts yang sudah dikurangi
    shapes = None
    for x,y in val_data.take(1):
        shapes = x.shape
    input_shape = shapes[1:]
    
    inputs = Input(shape=input_shape)
    
    # Augmentasi
    aug = tf.keras.Sequential([
        ZeroPadding2D(4),
        RandomCrop(32, 32),
        RandomFlip('horizontal')
    ], name='aug')
    x = aug(inputs)
    
    # Conv layers dengan filter counts yang sudah dikurangi
    x = QConv2DBatchnorm(
        actual_filters[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
        bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l2(1e-4),
        use_bias=True,
        name='fused_convbn_0'
    )(x)
    x = QActivation('quantized_relu(bits=12,integer=2)', name='conv_act_0')(x)
    x = MaxPooling2D()(x)
    
    x = QConv2DBatchnorm(
        actual_filters['1a'],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
        bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l2(1e-4),
        use_bias=True,
        name='fused_convbn_1a'
    )(x)
    x = QActivation('quantized_relu(bits=12,integer=2)', name='conv_act_1a')(x)
    x = MaxPooling2D()(x)
    
    x = QConv2DBatchnorm(
        actual_filters['1b'],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
        bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l2(1e-4),
        use_bias=True,
        name='fused_convbn_1b'
    )(x)
    x = QActivation('quantized_relu(bits=10,integer=2)', name='conv_act_1b')(x)
    x = MaxPooling2D()(x)
    
    if actual_filters['1c'] > 0:
        x = QConv2DBatchnorm(
            actual_filters['1c'],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
            bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l2(1e-4),
            use_bias=True,
            name='fused_convbn_1c'
        )(x)
        x = QActivation('quantized_relu(bits=10,integer=2)', name='conv_act_1c')(x)
        x = MaxPooling2D()(x)
    
    x = QConv2DBatchnorm(
        actual_filters[2],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_quantizer="quantized_bits(bits=9,integer=1,alpha=1)",
        bias_quantizer="quantized_bits(bits=12,integer=4,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l2(1e-4),
        use_bias=True,
        name='fused_convbn_2'
    )(x)
    x = QActivation('quantized_relu(bits=12,integer=3)', name='conv_act_2')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='conv_globalavg')(x)
    
    y = QDense(
        actual_filters[3],
        kernel_quantizer="quantized_bits(bits=12,integer=2,alpha=1)",
        kernel_regularizer=l2(1e-4),
        use_bias=False,
        name='dense_1'
    )(x)
    y = BatchNormalization(name='bn_3')(y)
    y = QActivation('quantized_relu(12,3)', name='dense_act_0')(y)
    y = Dropout(0.1, name='dense_dropout_0')(y)
    
    outputs = QDense(
        actual_filters[4],
        kernel_quantizer="quantized_bits(bits=16,integer=5,alpha=1)",
        bias_quantizer="quantized_bits(bits=16,integer=5,alpha=1)",
        kernel_regularizer=l2(1e-4),
        name='dense_2'
    )(y)
    
    model_full = Model(inputs, outputs, name="model_full_filterpruned")
    
    # Compile dan train
    model_full.compile(
        optimizer=tf.keras.optimizers.Adam(0.002),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]
    
    model_full.fit(
        train_data,
        epochs=120,
        verbose=2,
        validation_data=val_data,
        callbacks=callbacks
    )


# Build inference model (tanpa augmentasi dan dropout)
def build_inference_no_aug_softcoded(trained_model):
    layers_to_skip = {
        'aug',
        'dense_dropout_0',
        'input_1'
    }
    
    inp = tf.keras.Input(shape=trained_model.input_shape[1:], name='input_1')
    x = inp
    
    for layer in trained_model.layers:
        if layer.name in layers_to_skip:
            continue
        x = layer(x)
    
    return tf.keras.Model(inp, x, name='model_full_no_aug_softcoded')

model_full = build_inference_no_aug_softcoded(model_full)

# --- Testing ---
X_test, Y_test = [], []
for x, y in test_data.unbatch().take(10000):
    X_test.append(x.numpy())
    Y_test.append(y.numpy())
X_test = np.stack(X_test)
Y_test = np.stack(Y_test)

dense_out_logits = model_full.predict(X_test, batch_size=BATCH_SIZE)
probs = tf.nn.softmax(dense_out_logits).numpy()
pred_labels = np.argmax(probs, axis=1)
true_labels = Y_test
acc = accuracy_score(true_labels, pred_labels)
print(f"Akurasi model : {acc * 100:.2f}%")

np.save('npy/c10_X_test_main.npy', X_test)
np.save('npy/c10_Y_test_main.npy', Y_test)
np.save(f'keras/{suffix}/dense_out_logits_main.npy', dense_out_logits)

print("data test tersimpan")

model_full.save(model_name)
print(f"Model tersimpan di {model_name}")

with open(f'keras/{suffix}/model_full_summary.txt', 'w') as f:
    model_full.summary(print_fn=lambda x: f.write(x + '\n'))

akurasi_text = f"acc_{acc * 100:.2f}_filterpruned"
with open(f'keras/{suffix}/{akurasi_text}', 'w') as f:
    f.write(f"Filter Pruning Applied\n")
    f.write(f"Accuracy: {acc * 100:.2f}%\n")

print("\n\nls keras/")
[print(f) for f in listdir('keras')]
for k, v in vars(args).items():
    print(f"{k}: {v}")
