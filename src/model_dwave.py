"""
Class for estimating the model distribution of an RBM with dwave quantum annealing
"""

import json

import numpy as np
from networkx import Graph

from dwave.embedding.chimera import find_biclique_embedding
from dwave.embedding import embed_ising
from dwave_networkx import chimera_graph
from dimod.reference.samplers import SimulatedAnnealingSampler
from dwave.inspector import show
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from model import Model

class ModelDWAVE(Model):
    """
    Base class for model
    """
    def estimate_model(self, beta, plot_result=False):
        """
        Estimate the model distribution by sampling DWave 2000Q
        """
        num_reads = 200
        embedding = self.generate_embedding()
        d_sampler = DWaveSampler(solver={'topology__type': 'chimera'})
        emb_problem = FixedEmbeddingComposite(d_sampler, embedding)

        h_bias, j_coupling = self.generate_couplings(beta)

        s_pause = 0.4
        a_schedule = [
                [0.0, 0.0],
                [50.0, s_pause],
                [1050.0, s_pause],
                [1100.0, 1.0]
                ]

        response = emb_problem.sample_ising(
            h_bias,
            j_coupling,
            return_embedding=True,
            num_reads=num_reads,
            num_spin_reversal_transforms=5,
            anneal_schedule = a_schedule)

        if plot_result:
            show(response)

        vis_states, hid_states = self.extract_values_from_samples(response)

        return np.dot(vis_states.transpose(), hid_states) / num_reads

    def extract_values_from_samples(self, response):
        """
        Take a dwave sampleset and extract hidden and visible states from them
        """
        vis_states = np.zeros([len(response.samples()), len(self.visible)])
        hid_states = np.zeros([len(response.samples()), len(self.hidden)])

        for i in range(len(response.samples())):
            for vis_id in range(len(self.visible)):
                vis_states[i, vis_id] = response.samples()[i]['v_{0}'.format(vis_id)] / 2 + 0.5

            for hid_id in range(len(self.hidden)):
                hid_states[i, hid_id] = response.samples()[i]['h_{0}'.format(hid_id)] / 2 + 0.5

        return vis_states, hid_states

    def generate_embedding(self):
        """
        Generate dwave couplings for the RBM in question
        """
        v_labels = []
        h_labels = []

        for i in range(len(self.visible)):
            v_labels.append("v_{0}".format(i))

        for i in range(len(self.hidden)):
            h_labels.append("h_{0}".format(i))

        t_graph = self.generate_graph()
        vis, hid = find_biclique_embedding(v_labels, h_labels, 16, target_edges=t_graph.edges)

        return vis | hid

    def generate_couplings(self, beta):
        """
        Generate coupling parameters h and J
        """
        j_couplings = {}
        h_mag = {}

        for i in range(len(self.visible)):
            for j in range(len(self.hidden)):
                j_couplings[('v_{0}'.format(i), 'h_{0}'.format(j))] = -self.weights[i, j] / 4

        for i in range(len(self.visible)):
            h_mag['v_{0}'.format(i)] = -self.visible[i] / 2

            for j in range(len(self.hidden)):
                h_mag['v_{0}'.format(i)] -= self.weights[i, j] / 4

        for j in range(len(self.hidden)):
            h_mag['h_{0}'.format(j)] = -self.hidden[j] / 2

            for i in range(len(self.hidden)):
                h_mag['h_{0}'.format(j)] -= self.weights[i, j] / 4

        for h_key in h_mag.keys():
           h_mag[h_key] /= beta

        for j_key in j_couplings.keys():
            j_couplings[j_key] /= beta

        return h_mag, j_couplings

    def generate_graph(self):
        """
        Generate the chimera graph for DWave 2000Q
        """
        with open('dwave2000q_6.json') as json_file:
            graph_json = json.load(json_file)

        dwave_6 = Graph()

        dwave_6.add_nodes_from(graph_json['qubits'])
        dwave_6.add_edges_from(graph_json['couplings'])

        return dwave_6
