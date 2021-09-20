"""
This program is supposed to test and compare different sampling approaches
for Boltzmann machines.


TODO:
    [X] - Computing <vh>_model analytically
    [X] - Computing <vh>_model using Contrastive Divergence
    [ ] - Computing <vh>_model using SQA
    [ ] - Computing <vh>_model using DWave quantum annealing
    [ ] - Plotting results
"""

import numpy as np

from dataset import Dataset
from utils import l1_between_models
from model import RBMParameters
from model_analytical import ModelAnalytical
from model_cd import ModelCD


def run():
    n_samples = 1000
    n_size = 8
    seed = 3104804

    generator = np.random.default_rng(seed)

    dataset = Dataset(generator, n_size, n_samples)
    parameters = RBMParameters(n_size, n_size, generator)

    analytical_model = ModelAnalytical(parameters)
    cd1_model = ModelCD(parameters, 1)
    cd5_model = ModelCD(parameters, 5)
    cd25_model = ModelCD(parameters, 25)
    cd100_model = ModelCD(parameters, 100)
    cd1000_model = ModelCD(parameters, 1000)

    cd_1_avg = 0.0
    cd_5_avg = 0.0
    cd_25_avg = 0.0
    cd_100_avg = 0.0
    cd_1000_avg = 0.0

    results_analytical = analytical_model.estimate_model()

    results_cd1 = cd1_model.estimate_model(dataset)
    results_cd5 = cd5_model.estimate_model(dataset)
    results_cd25 = cd25_model.estimate_model(dataset)
    results_cd100 = cd100_model.estimate_model(dataset)
    results_cd1000 = cd1000_model.estimate_model(dataset)

    cd_1_avg += l1_between_models(results_analytical, results_cd1)
    cd_5_avg += l1_between_models(results_analytical, results_cd5)
    cd_25_avg += l1_between_models(results_analytical, results_cd25)
    cd_100_avg += l1_between_models(results_analytical, results_cd100)
    cd_1000_avg += l1_between_models(results_analytical, results_cd1000)

    print("CD1")
    print(cd_1_avg / 100)
    print("CD5")
    print(cd_5_avg / 100)
    print("CD25")
    print(cd_25_avg / 100)
    print("CD100")
    print(cd_100_avg / 100)
    print("CD1000")
    print(cd_1000_avg / 100)


if __name__ == "__main__":
    run()
