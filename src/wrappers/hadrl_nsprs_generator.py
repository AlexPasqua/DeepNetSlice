import copy
import random
import warnings
from collections import Counter
from typing import Tuple, Union, List

import gym
import networkx as nx
import numpy as np


class NSPRsGeneratorHADRL(gym.Wrapper):
    """
    Wrapper to make the simulator generate data the same way as in the
    paper HA-DRL[1].

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            env: gym.Env,
            nsprs_per_ep: int = 5,
            vnfs_per_nspr: int = 5,
            cpu_req_per_vnf: int = 25,
            ram_req_per_vnf: int = 150,
            bw_req_per_vl: int = 2000,
            load: float = 0.5,
    ):
        super().__init__(env)
        if self.env.nsprs_per_episode is not None:
            warnings.warn("The environment already has a fixed number of NSPRs"
                          "per episode. The wrapper will override this value.")
        self.nsprs_per_ep = nsprs_per_ep
        self.vnfs_per_nspr = vnfs_per_nspr
        self.cpu_req_per_vnf = cpu_req_per_vnf
        self.ram_req_per_vnf = ram_req_per_vnf
        self.bw_req_per_vl = bw_req_per_vl
        self.load = load
        self.tot_cpu_cap = self._get_tot_cpu_cap()
        self.nspr_model = self._get_nspr_model()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.unwrapped.nsprs = self._generate_nsprs()
        return obs

    def _get_nspr_model(self):
        nspr_model = nx.DiGraph()
        nspr_model.add_node(0, reqCPU=self.cpu_req_per_vnf, reqRAM=self.ram_req_per_vnf)
        for i in range(1, self.vnfs_per_nspr):
            nspr_model.add_edge(i-1, i, reqBW=self.bw_req_per_vl)
            nspr_model.add_node(i, reqCPU=self.cpu_req_per_vnf, reqRAM=self.ram_req_per_vnf)
        return nspr_model

    def _generate_nsprs(self):
        # # if env wrapped in TimeLimit, max arrival time of NSPRs is max episode
        # # steps minus the number of VNFs in the NSPRs (to give them time to be
        # # placed), otherwise it's a default value
        # max_time = self.env.time_step + \
        #            self.env.get('_max_episode_steps', 100) - self.vnfs_per_nspr
        # arr_times = random.sample(range(self.env.time_step, max_time),
        #                               self.nsprs_per_ep)

        try:
            nsprs_duration = self.env._max_episode_steps
        except AttributeError:
            nsprs_duration = 100
        # computed according to Sec. VII.C of HA-DRL paper
        arr_rate = self.load * self.tot_cpu_cap * (1 / nsprs_duration) / self.cpu_req_per_vnf
        arr_times = np.random.poisson(lam=arr_rate, size=self.nsprs_per_ep)
        arr_times.sort()
        occurrences_arr_times = Counter(arr_times)
        nsprs_dict = {}
        for arr_time, occurrences in occurrences_arr_times.items():
            nsprs_dict[arr_time] = [copy.deepcopy(self.nspr_model) for _ in range(occurrences)]
        return nsprs_dict

    def _get_tot_cpu_cap(self):
        tot_cpu_cap = 0
        for node_id in self.env.psn.nodes:
            node = self.env.psn.nodes[node_id]
            if node['NodeType'] == 'server':
                tot_cpu_cap += node['CPUcap']
        return tot_cpu_cap
