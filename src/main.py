"""
This program is supposed to test and compare different sampling approaches
for Boltzmann machines.


TODO:
    [X] - Computing <vh>_model analytically
    [X] - Computing <vh>_model using Contrastive Divergence
    [X] - Computing <vh>_model using SQA
        -> Didn't work properly. Might check for parameters later
    [X] - Computing <vh>_model using DWave quantum annealing
    [ ] - Plotting results
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

from dataset import Dataset
from utils import l1_between_models
from model import RBMParameters
from model_analytical import ModelAnalytical
from model_cd import ModelCD
from model_dwave import ModelDWAVE

def test_working_models():
    """
    Function for testing and comparing models that
    have been confirmed to work.
    """
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

def run():
    """
    Basic run script of the program
    """
    n_size = 60

    params = RBMParameters(n_size, n_size, np.random.default_rng(1230))
    dataset = Dataset(np.random.default_rng(120), n_size, 1000)

    model_dwave = ModelDWAVE(params)
    model_analytical = ModelCD(params, 1000)
    model_cd1 = ModelCD(params, 1)
    model_cd10 = ModelCD(params, 10)
#    model_analytical = ModelAnalytical(params)

    an_res = model_analytical.estimate_model(dataset)
    betas = [1.0, 2.0, 3.0, 4.0, 4.5]

    for b in betas:
        res_1 = model_dwave.estimate_model(b)
        res_2 = model_cd1.estimate_model(dataset)
        res_3 = model_cd10.estimate_model(dataset)
        print("l1 for beta: {0}".format(b))
        print(l1_between_models(an_res, res_1))
        print("l1 for the cd1")
        print(l1_between_models(an_res, res_2))
        print("l1 for the cd10")
        print(l1_between_models(an_res, res_3))
        print("l1 for the cd1000")

if __name__ == "__main__":
    run()
