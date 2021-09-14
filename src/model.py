"""
Root class of the model objects. Used for passing parameters around other models.
"""

import numpy as np

class Model():
    """
    Base class for model
    """
    def __init__(self, visible_size, hidden_size, generator):
        self.weights = generator.normal(0.0, 0.01, [visible_size, hidden_size]) #is the range 0.0 and 0.01 good enough? Should it be higher
        self.visible = np.zeros(visible_size, dtype=float)
        self.hidden = np.zeros(hidden_size, dtype=float)

    def estimate_model(self):
        raise Exception("This model class cannot be used to estimate model distribution!!")
