"""
This program is supposed to test and compare different sampling approaches
for boltzman machines.


TODO:
    [ ] - Computing <vh>_model analytically
    [ ] - Computing <vh>_model using Contrastive Divergence
    [ ] - Computing <vh>_model using SQA
    [ ] - Computing <vh>_model using DWave quantum annealing
    [ ] - Plotting results
"""

import numpy as np

from dataset import Dataset
from model import Model
from model_analytical import ModelAnalytical

def run():
    n_samples = 1000
    n_size = 8
    seed = 3104802
    generator = np.random.default_rng(seed)

    dataset = Dataset(generator, n_size, n_samples)
    model = Model(n_size, n_size, generator)
    
    analytical_model = ModelAnalytical(model)
    print(analytical_model.estimate_model(dataset))

if __name__ == "__main__":
    run()
