"""
Root class of the model objects. Used for passing parameters around other models.
"""

import numpy as np

class Model:
    """
    Base class for model
    """
    def __init__(self, parameters):
        self.weights = parameters.weights
        self.visible = parameters.visible
        self.hidden = parameters.hidden

    def estimate_model(self):
        raise Exception("This model class cannot be used to estimate model distribution!!")

class RBMParameters:
    """
    Base class for model parameters
    """
    def __init__(self, visible_size, hidden_size, generator):
        self.weights = generator.normal(0.2, 0.5, [visible_size, hidden_size])
        self.visible = generator.normal(0.0, 0.1, visible_size)
        self.hidden = generator.normal(0.0, 0.1, hidden_size)