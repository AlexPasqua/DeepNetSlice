import copy
import math
import warnings

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
        if nsprs_per_ep is None:
            # no limit, just use max steps (if not None), otherwise infinite episode
            nsprs_per_ep = math.inf
        self.unwrapped.nsprs_per_episode = nsprs_per_ep
        self.nsprs_per_ep = nsprs_per_ep
        self.vnfs_per_nspr = vnfs_per_nspr
        self.cpu_req_per_vnf = cpu_req_per_vnf
        self.ram_req_per_vnf = ram_req_per_vnf
        self.bw_req_per_vl = bw_req_per_vl
        self.load = load
        self.tot_cpu_cap = self._get_tot_cpu_cap()
        self.nspr_model = self._get_nspr_model()
        self.max_steps = None
        try:
            # if env is wrapped in TimeLimit, max arrival time of NSPRs is max episode length
            self.max_steps = self.env._max_episode_steps
            self.nsprs_duration = min(self.max_steps, 100)
        except AttributeError or TypeError:
            self.nsprs_duration = 100
        # computed according to Sec. VII.C of HA-DRL paper
        self.arr_rate = self.load * self.tot_cpu_cap / self.nsprs_duration / self.cpu_req_per_vnf

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.unwrapped.nsprs = self._generate_nsprs()
        return obs

    def _get_nspr_model(self):
        nspr_model = nx.DiGraph()
        nspr_model.add_node(0, reqCPU=self.cpu_req_per_vnf,
                            reqRAM=self.ram_req_per_vnf, placed=-1)
        for i in range(1, self.vnfs_per_nspr):
            nspr_model.add_edge(i - 1, i, reqBW=self.bw_req_per_vl, placed=[])
            nspr_model.add_node(i, reqCPU=self.cpu_req_per_vnf,
                                reqRAM=self.ram_req_per_vnf, placed=-1)
        return nspr_model

    def _generate_nsprs(self):
        if self.arr_rate >= 0.3:
            nsprs_dict = self._generate_nsprs_poisson()
        else:
            nsprs_dict = self._generate_nsprs_deterministic()
        return nsprs_dict

    def _generate_nsprs_poisson(self):
        cur_arr_time = self.env.time_step
        created_nsprs = 0
        nsprs_dict = {}
        while True:
            # NOTE: if self.max_steps is None, and the poisson sampling keeps
            # generating 0, this will loop forever, but since this is executed
            # only for a sufficiently high arrival rate, this is extremely unlikely to happen
            poisson_sample = np.random.poisson(lam=self.arr_rate)
            if poisson_sample > 0:
                cur_nspr = copy.deepcopy(self.nspr_model)
                cur_nspr.graph['ArrivalTime'] = cur_arr_time
                cur_nspr.graph['DepartureTime'] = cur_arr_time + self.nsprs_duration
                nsprs_to_create = min(poisson_sample, self.nsprs_per_ep - created_nsprs)
                if nsprs_to_create <= 0:
                    break
                nsprs_dict[cur_arr_time] = [copy.deepcopy(cur_nspr) for _ in range(nsprs_to_create)]
                created_nsprs += nsprs_to_create
            cur_arr_time += 1
            if self.max_steps is not None and self.env.time_step + cur_arr_time > self.max_steps:
                break
        return nsprs_dict

    def _generate_nsprs_deterministic(self):
        if self.arr_rate >= 1:
            raise NotImplementedError
            # this function is called only for low arrival rates
        else:
            one_every_how_many_steps = round(1 / self.arr_rate)
            # decimal_part = round(one_every_how_many_steps - int(one_every_how_many_steps), 2)
            # one_every_how_many_steps = int(one_every_how_many_steps)
            # correction_every_how_many_steps = round(1 / decimal_part)
            nsprs_dict = {}
            step = self.env.time_step
            # steps_without_correction = 0
            created_nsprs = 0
            while True:
                if step % one_every_how_many_steps == 0:
                    cur_nspr = copy.deepcopy(self.nspr_model)
                    cur_nspr.graph['ArrivalTime'] = step
                    cur_nspr.graph['DepartureTime'] = step + self.nsprs_duration
                    nsprs_dict[step] = [cur_nspr]
                    created_nsprs += 1
                    # if step % one_every_how_many_steps == 0 and \
                    #         steps_without_correction == correction_every_how_many_steps:
                    #     nsprs_dict[step].append(copy.deepcopy(cur_nspr))
                    #     created_nsprs += 1
                    # if steps_without_correction == correction_every_how_many_steps:
                    #     steps_without_correction = 0
                step += 1
                # steps_without_correction += 1
                if created_nsprs >= self.nsprs_per_ep or \
                        (self.max_steps is not None and step > self.max_steps):
                    break
            return nsprs_dict

    def _get_tot_cpu_cap(self):
        tot_cpu_cap = 0
        for node_id in self.env.psn.nodes:
            node = self.env.psn.nodes[node_id]
            if node['NodeType'] == 'server':
                tot_cpu_cap += node['CPUcap']
        return tot_cpu_cap
