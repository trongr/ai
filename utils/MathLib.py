import numpy as np
import tensorflow as tf


def log(x):
    return tf.log(x + 1e-8)


def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
