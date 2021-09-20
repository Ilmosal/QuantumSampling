"""
Utility functions go here
"""

import numpy as np


def unpackbits(val, num_bits):
    val = np.array([val])
    valshape = list(val.shape)
    val = val.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=val.dtype).reshape([1, num_bits])
    return (val & mask).astype(bool).astype(int).reshape(valshape + [num_bits])

def sample(val):
    return (val > np.random.uniform(0.0, 1.0, val.shape)).astype(float)


def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))


def l1_between_models(base_model, estimate_model):
    return np.sum(np.abs(base_model - estimate_model))
