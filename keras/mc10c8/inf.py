import tensorflow as tf
import numpy as np


from tensorflow.keras.models import Model
from qkeras import QActivation, QDense, QConv2DBatchnorm
from tensorflow.keras.utils import custom_object_scope
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude

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

model = tf.keras.models.load_model('main_full.keras', custom_objects=custom_objects, compile=True)
X_test = np.load('X_test_main.npy')
Y_test = np.load('Y_test_main.npy')
output = model.predict(X_test[:20])
labels = np.argmax(output, axis=1)

for o in range(len(labels)):
    print(f"{labels[o]} vs {Y_test[o]}. output: {output[o]}")