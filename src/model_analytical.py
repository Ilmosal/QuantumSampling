"""
Root class of the model objects. Used for passing parameters around other models.
"""

import numpy as np

class ModelAnalytical():
    """
    Base class for model
    """
    def __init__(self, model):
        self.weights = model.weights
        self.visible = model.visible
        self.hidden = model.hidden

        self.unpack_types = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32
        }

    def estimate_model(self, data):
        vh = None
        Z = self.compute_partition_function()

        for v_val in range(2**len(self.visible)):
            
        
        return vh

    def get_int_format(self):
        return self.unpack_types[len(self.visible)]

    def compute_pv(self, v):
        return 0

    def compute_partition_function(self):
        Z = 0

        for i in range(2**len(self.visible)):
            for j in range(2**len(self.hidden)):
                v_val = np.uint32(i).astype(self.get_int_format())
                h_val = np.uint32(j).astype(self.get_int_format())

                vis_ar = np.matrix(np.unpackbits(v_val))
                hid_ar = np.matrix(np.unpackbits(h_val))

                E = - np.sum(vis_ar.T * hid_ar * self.weights) - self.visible * vis_ar.T - self.hidden * hid_ar.T

                Z += np.exp(E)

        return Z

