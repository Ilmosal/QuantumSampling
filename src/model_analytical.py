"""
Class for computing the model distribution of a RBM analytically . Used for passing parameters around other models.
"""

import numpy as np

from model import Model
from utils import unpackbits

class ModelAnalytical(Model):
    """
    Base class for model
    """

    def __init__(self, parameters):
        super(ModelAnalytical, self).__init__(parameters)

        self.unpack_types = {
            8: 8,
            10: 10,
            16: 16,
            32: 32
        }

    def estimate_model(self):
        vh = np.zeros([len(self.visible), len(self.hidden)])
        z = self.compute_partition_function()

        for v in range(2 ** len(self.visible)):
            vis_ar = np.matrix(unpackbits(v, self.get_int_format()))
            hid_ar = self.compute_hid_ar(vis_ar)
            vh += vis_ar.T * hid_ar * self.compute_pv(vis_ar, z)

        return vh

    def total_pv(self):
        total = 0.0

        z = self.compute_partition_function()

        for v in range(2 ** len(self.visible)):
            vis_ar = np.matrix(unpackbits(v, self.get_int_format()))
            total += self.compute_pv(vis_ar, z)

        return total

    def get_int_format(self):
        return self.unpack_types[len(self.visible)]

    def compute_hid_ar(self, vis_ar):
        return 1.0 / (1.0 + np.exp(-self.hidden - np.dot(vis_ar, self.weights)))

    def compute_pv(self, v, z):
        pv = 0.0

        for h in range(2 ** len(self.hidden)):
            hid_ar = np.matrix(unpackbits(h, self.get_int_format()))

            pv += np.exp(np.sum(np.multiply(v.T * hid_ar, self.weights)) + self.visible * v.T + self.hidden * hid_ar.T)

        return (pv / z)[0, 0]

    def compute_partition_function(self):
        z = 0

        for i in range(2 ** len(self.visible)):
            for j in range(2 ** len(self.hidden)):
                vis_ar = np.matrix(unpackbits(i, self.get_int_format()))
                hid_ar = np.matrix(unpackbits(j, self.get_int_format()))

                z += np.exp(np.sum(np.multiply(vis_ar.T * hid_ar, self.weights)) + self.visible * vis_ar.T + self.hidden * hid_ar.T)

        return z[0, 0]
