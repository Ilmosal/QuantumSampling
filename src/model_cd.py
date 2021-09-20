"""
Class for estimating the model distribution of an RBM with contrastive divergence algorithm.
"""

import numpy as np

from model import Model
from utils import sample, sigmoid

class ModelCD(Model):
    """
    Base class for model
    """

    def __init__(self, parameters, cd_iter, seed=None):
        super(ModelCD, self).__init__(parameters)

        self.cd_iter = cd_iter
        self.generator = np.random.default_rng(seed)

    def estimate_model(self, dataset):
        vis_state = np.copy(dataset.get_data())
        hid_state = self.activate_hidden(vis_state)

        for i in range(self.cd_iter):
            vis_state = self.activate_visible(hid_state)
            hid_state = self.activate_hidden(vis_state)

        vh_cd = np.dot(vis_state.transpose(), hid_state) / len(dataset.get_data())

        return vh_cd

    def activate_hidden(self, values):
        return sample(sigmoid(np.dot(values, self.weights) + self.hidden))

    def activate_visible(self, values):
        return sample(sigmoid(np.dot(values, self.weights.transpose()) + self.visible))
