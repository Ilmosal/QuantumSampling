"""
Class for computing the model distribution of a RBM analytically.
"""

import numpy as np

from model import Model
from utils import unpackbits

class ModelAnalytical(Model):
    """
    Base class for model
    """
    def estimate_model(self):
        v_h = np.zeros([len(self.visible), len(self.hidden)])
        z_part = self.compute_partition_function()

        for vis in range(2 ** len(self.visible)):
            vis_ar = np.matrix(unpackbits(vis, self.get_int_format()))
            hid_ar = self.compute_hid_ar(vis_ar)
            v_h += vis_ar.T * hid_ar * self.compute_pv(vis_ar, z_part)

        return v_h

    def compute_min_energy(self):
        """
        Compute and return the minimum energy configuration of the model
        """
        min_energy = 0.0
        min_configuration = [None, None]

        for i in range(2 ** len(self.visible)):
            for j in range(2 ** len(self.hidden)):
                vis_ar = np.matrix(unpackbits(i, self.get_int_format()))
                hid_ar = np.matrix(unpackbits(j, self.get_int_format()))

                tot_energy = self.compute_energy(vis_ar, hid_ar)

                if tot_energy < min_energy:
                    min_configuration[0] = vis_ar
                    min_configuration[1] = hid_ar
                    min_energy = tot_energy

        return min_energy, min_configuration

    def compute_energy(self, vis_ar, hid_ar):
        """
        Compute the total energy of the configuration
        """
        return -np.sum(np.multiply(vis_ar.T * hid_ar, self.weights)) - self.visible * vis_ar.T - self.hidden * hid_ar.T

    def total_pv(self):
        """
        Compute total prob of v over all possible v. Test function that should always be 1.
        """
        total = 0.0

        z_part = self.compute_partition_function()

        for vis in range(2 ** len(self.visible)):
            vis_ar = np.matrix(unpackbits(vis, self.get_int_format()))
            total += self.compute_pv(vis_ar, z_part)

        return total

    def get_int_format(self):
        """
        Get int format for unpacking
        """
        return len(self.visible)

    def compute_hid_ar(self, vis_ar):
        """
        Compute the probabilities of the hidden array from vis_ar
        """
        return 1.0 / (1.0 + np.exp(-self.hidden - np.dot(vis_ar, self.weights)))

    def compute_pv(self, vis_ar, z_part):
        """
        Compute the probability of vis_ar
        """
        p_v = 0.0

        for hid in range(2 ** len(self.hidden)):
            hid_ar = np.matrix(unpackbits(hid, self.get_int_format()))

            p_v += np.exp(np.sum(np.multiply(vis_ar.T * hid_ar, self.weights)) + self.visible * vis_ar.T + self.hidden * hid_ar.T)

        return (p_v / z_part)[0, 0]

    def compute_partition_function(self):
        """
        Compute the partition function for the model
        """
        z_part = 0

        for i in range(2 ** len(self.visible)):
            for j in range(2 ** len(self.hidden)):
                vis_ar = np.matrix(unpackbits(i, self.get_int_format()))
                hid_ar = np.matrix(unpackbits(j, self.get_int_format()))

                z_part += np.exp(np.sum(np.multiply(vis_ar.T * hid_ar, self.weights)) + self.visible * vis_ar.T + self.hidden * hid_ar.T)

        return z_part[0, 0]
