from keras import backend as K
def some_function(x):
    return(x)
def my_function(x):
    x = K.some_function(x)
    return x




import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import glob

def my_RMSE(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = y_pred - y_true
    return K.sqrt(K.mean(K.square(tf.norm(diff, ord='euclidean', axis=-1)), axis=-1))


def mse_and_norm_mse(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = y_pred - y_true
    y_pred_norm = tf.norm(y_pred, ord='euclidean', axis=-1)
    y_true_norm = tf.norm(y_true, ord='euclidean', axis=-1)
    norm_diff = y_pred_norm - y_true_norm
    loss = K.mean(K.square(tf.norm(diff, ord='euclidean', axis=-1)), axis=-1) + K.mean(K.square(norm_diff), axis=-1)
import keras.backend as K
def exp(x):
    return K.exp(-K.pow(x,1))

def gaussian(x):
    return K.exp(-K.pow(x,2))

def gaussian_4(x):
    return K.exp(-K.pow(x,4))

