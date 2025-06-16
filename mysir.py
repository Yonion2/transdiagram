from mydiffusion import DiffusionModel
import numpy as np
import future.utils
from copy import deepcopy
from collections import defaultdict
import gc
import random
__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class SIRModel(DiffusionModel):
    """
       Model Parameters to be specified via ModelConfig

       :param beta: The infection rate (float value in [0,1])
       :param gamma: The recovery rate (float value in [0,1])
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor

             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {
            "3": 3,
            "2": 2,
            "1": 1,
            "-1":-1,
            "-2":-2,
            "-3":-3
        }

        self.parameters = {
            "model": {
                "gamma_1m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m1m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m1m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m2m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m2m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m2m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m3m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m3m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_1m3m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],

                    "optional": False},
                "gamma_2m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m1m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m1m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m2m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m2m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m2m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m3m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m3m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_2m3m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m1m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m1m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m2m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m2m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m2m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m3m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m3m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_3m3m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m1m2m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m1m2m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m1m3m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m1m3m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m2m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m2m1m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m2m1m3": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m2m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m3m1m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m3m1m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m3m2m1": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma_m3m2m2": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "p_12": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_13": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_23": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_21": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_23": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_31": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_32": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m11": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m12": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m13": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m21": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m22": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m23": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m31": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m32": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m33": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m1m2": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m1m3": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m2m1": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m2m3": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m3m1": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "p_m3m2": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False},
                "tp_rate": {
                    "descr": "Whether if the infection rate depends on the number of infected neighbors",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                }
            },
            "nodes": {},
            "edges": {},
        }

        self.name = "multipleState"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(list(self.available_statuses.values()))

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        states = [[3,2,1], [3,2,1,-1,-2,-3],[3,2,1,-2,-3],[3,2,1,-1,-3],[3,2,1,-1,-2]]
        # 3，2，1，-1，-2，-3的转移
        probs = [[1- self.params['model']['p_32'] - self.params['model']['p_31'], self.params['model']['p_32'], self.params['model']['p_31']],
                 [self.params['model']['p_23'], 1- self.params['model']['p_21'] - self.params['model']['p_23'], self.params['model']['p_21']],
                 [self.params['model']['p_13'], self.params['model']['p_12'], 1- self.params['model']['p_12'] - self.params['model']['p_13']],
                 [self.params['model']['p_m13'], self.params['model']['p_m12'], self.params['model']['p_m11'], 1- self.params['model']['p_m1m2'] - self.params['model']['p_m1m3'] - self.params['model']['p_m11'] - self.params['model']['p_m12'] - self.params['model']['p_m13'], self.params['model']['p_m1m2'], self.params['model']['p_m1m3']],
                 [self.params['model']['p_m23'], self.params['model']['p_m22'], self.params['model']['p_m21'], self.params['model']['p_m2m1'], 1- self.params['model']['p_m2m1'] - self.params['model']['p_m2m3'] - self.params['model']['p_m21'] - self.params['model']['p_m22'] - self.params['model']['p_m23'], self.params['model']['p_m2m3']],
                 [self.params['model']['p_m33'], self.params['model']['p_m32'], self.params['model']['p_m31'], self.params['model']['p_m3m1'], self.params['model']['p_m3m2'], 1- self.params['model']['p_m3m1'] - self.params['model']['p_m3m2'] - self.params['model']['p_m31'] - self.params['model']['p_m32'] - self.params['model']['p_m33'],]]
        
        # m1 m2 m3的传染
        infection_rates = [[[1-self.params['model']['gamma_3m1m1']-self.params['model']['gamma_3m1m2']-self.params['model']['gamma_3m1m3'], 0, 0, self.params['model']['gamma_3m1m1'], self.params['model']['gamma_3m1m2'], self.params['model']['gamma_3m1m3']],
                            [0, 1-self.params['model']['gamma_2m1m1']-self.params['model']['gamma_2m1m2']-self.params['model']['gamma_2m1m3'], 0, self.params['model']['gamma_2m1m1'], self.params['model']['gamma_2m1m2'], self.params['model']['gamma_2m1m3']],
                            [0, 0, 1-self.params['model']['gamma_1m1m1']-self.params['model']['gamma_1m1m2']-self.params['model']['gamma_1m1m3'], self.params['model']['gamma_1m1m1'], self.params['model']['gamma_1m1m2'], self.params['model']['gamma_1m1m3']],
                            [0, 0, 0, self.params['model']['gamma_m2m1m1'], 1 - self.params['model']['gamma_m2m1m1'] - self.params['model']['gamma_m2m1m3'], self.params['model']['gamma_m2m1m3']],
                            [0, 0, 0, self.params['model']['gamma_m3m1m1'], self.params['model']['gamma_m3m1m2'], 1 - self.params['model']['gamma_m3m1m1'] - self.params['model']['gamma_m3m1m2'], ]],
                           [[1-self.params['model']['gamma_3m2m1']-self.params['model']['gamma_3m2m2']-self.params['model']['gamma_3m2m3'], 0, 0, self.params['model']['gamma_3m2m1'], self.params['model']['gamma_3m2m2'], self.params['model']['gamma_3m2m3']],
                            [0, 1-self.params['model']['gamma_2m2m1']-self.params['model']['gamma_2m2m2']-self.params['model']['gamma_2m2m3'], 0, self.params['model']['gamma_2m2m1'], self.params['model']['gamma_2m2m2'], self.params['model']['gamma_2m2m3']],
                            [0, 0, 1-self.params['model']['gamma_1m2m1']-self.params['model']['gamma_1m2m2']-self.params['model']['gamma_1m2m3'], self.params['model']['gamma_1m2m1'], self.params['model']['gamma_1m2m2'], self.params['model']['gamma_1m2m3']],
                            [0, 0, 0, 1 - self.params['model']['gamma_m1m2m2'] - self.params['model']['gamma_m1m2m3'], self.params['model']['gamma_m1m2m2'], self.params['model']['gamma_m1m2m3']],
                            [0, 0, 0, self.params['model']['gamma_m3m2m1'] , self.params['model']['gamma_m3m2m2'], 1 - self.params['model']['gamma_m3m1m1'] - self.params['model']['gamma_m3m1m2']]],
                           [[1-self.params['model']['gamma_3m3m1']-self.params['model']['gamma_3m3m2']-self.params['model']['gamma_3m3m3'], 0, 0, self.params['model']['gamma_3m3m1'], self.params['model']['gamma_3m3m2'], self.params['model']['gamma_3m3m3']],
                            [0, 1-self.params['model']['gamma_2m3m1']-self.params['model']['gamma_2m3m2']-self.params['model']['gamma_2m3m3'], 0, self.params['model']['gamma_2m3m1'], self.params['model']['gamma_2m3m2'], self.params['model']['gamma_2m3m3']],
                            [0, 0, 1-self.params['model']['gamma_1m3m1']-self.params['model']['gamma_1m3m2']-self.params['model']['gamma_1m3m3'], self.params['model']['gamma_1m3m1'], self.params['model']['gamma_1m3m2'], self.params['model']['gamma_1m3m3']],
                            [0, 0, 0, 1 - self.params['model']['gamma_m1m3m2'] - self.params['model']['gamma_m1m3m3'], self.params['model']['gamma_m1m3m2'], self.params['model']['gamma_m1m3m3']],
                            [0, 0, 0, self.params['model']['gamma_m2m3m1'], 1 - self.params['model']['gamma_m2m3m1'] - self.params['model']['gamma_m2m3m3'], self.params['model']['gamma_m2m3m3']]]]
        random.shuffle(self.active)
        for u in self.active:
            u_status = self.status[u]
            if actual_status[u] == u_status:
                susceptible_neighbors = defaultdict(list)
                if self.graph.directed:
                    for v in self.graph.successors(u):
                        susceptible_neighbors[self.status[v]].append(v)
                else:
                    for v in self.graph.neighbors(u):
                        susceptible_neighbors[self.status[v]].append(v)
                if u_status == 3:
                    if actual_status[u] == u_status:
                        prob = probs[0]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[0][eventp]
                elif u_status == 2:
                    if actual_status[u] == u_status:
                        prob = probs[1]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[0][eventp]
                elif u_status == 1:
                    if actual_status[u] == u_status:
                        prob = probs[2]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[0][eventp]
                elif u_status == -1:
                    for ord, sta_temp in enumerate(states[2]):
                        if susceptible_neighbors[sta_temp]:
                            for v in susceptible_neighbors[sta_temp]:
                            # if actual_status[v] == sta_temp:
                                infectp = np.nonzero(np.random.multinomial(1, infection_rates[0][ord]))[0][0]
                                if states[1][infectp]<0:
                                    actual_status[v] = states[1][infectp]
                    if actual_status[u] == u_status:
                        prob = probs[3]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[1][eventp]

                elif u_status == -2:
                    for ord, sta_temp in enumerate(states[3]):
                        if susceptible_neighbors[sta_temp]:
                            for v in susceptible_neighbors[sta_temp]:
                                # if actual_status[v] == sta_temp:
                                infectp = np.nonzero(np.random.multinomial(1, infection_rates[1][ord]))[0][0]
                                if states[1][infectp]<0:
                                    actual_status[v] = states[1][infectp]
                    if actual_status[u] == u_status:
                        prob = probs[4]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[1][eventp]
                elif u_status == -3:
                    for ord, sta_temp in enumerate(states[4]):
                        if susceptible_neighbors[sta_temp]:
                            for v in susceptible_neighbors[sta_temp]:
                                # if actual_status[v] == sta_temp:
                                infectp = np.nonzero(np.random.multinomial(1, infection_rates[2][ord]))[0][0]
                                if states[1][infectp]<0:
                                    actual_status[v] = states[1][infectp]
                    if actual_status[u] == u_status:
                        prob = probs[5]
                        eventp = np.nonzero(np.random.multinomial(1, prob))[0][0]
                        actual_status[u] = states[1][eventp]

        delta, node_count, status_delta = self.status_delta(actual_status)
        del self.status
        gc.collect()
        self.status = deepcopy(actual_status)
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

