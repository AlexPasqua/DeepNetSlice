from abc import ABC, abstractmethod
from typing import Union

import gym
import numpy as np


class ResetWithLoad(gym.Wrapper, ABC):
    """ Abstract class. Wrapper to reset the PSN with a certain load """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.cpu_load = self.ram_load = self.bw_load = 0.

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self._init_psn_load()
        obs = self.env.get_observation()    # the obs in the env.reset method is outdated
        return obs

    def _init_psn_load(self):
        """ Initialize the PSN's load with the specified values """
        for _, node in self.env.psn.nodes.items():
            if node['NodeType'] == "server":
                node['availCPU'] = int(node['CPUcap'] * (1 - self.cpu_load))
                node['availRAM'] = int(node['RAMcap'] * (1 - self.ram_load))
        for _, link in self.env.psn.edges.items():
            link['availBW'] = int(link['BWcap'] * (1 - self.bw_load))


class ResetWithFixedLoad(ResetWithLoad):
    """ Reset the PSN with a certain - fixed - amount of load """

    def __init__(self, env: gym.Env, reset_load_perc: Union[float, dict] = 0.):
        """ Constructor

        :param env: :param env: the environment to wrap
        :param reset_load_perc: init percentage of load of the PSN's resources at each reset:
            if float, that value applies to all the resources for all nodes and links;
            if dict, it can specify the load for each type of resource.
        """
        super().__init__(env)
        assert isinstance(reset_load_perc, (float, dict))
        # define the load percentages of each resource
        if isinstance(reset_load_perc, float):
            assert 0 <= reset_load_perc <= 1
            self.cpu_load = self.ram_load = self.bw_load = reset_load_perc
        else:
            self.cpu_load = reset_load_perc.get('availCPU', 0)
            self.ram_load = reset_load_perc.get('availRAM', 0)
            self.bw_load = reset_load_perc.get('availBW', 0)
            assert 0 <= self.cpu_load <= 1 and 0 <= self.ram_load <= 1 and 0 <= self.bw_load <= 1


class ResetWithRandLoad(ResetWithLoad):
    """ Reset the PSN with a random uniform amount of load """

    def __init__(self, env: gym.Env, min_perc: Union[float, dict],
                 max_perc: Union[float, dict], same_for_all: bool = True):
        """ Constructor

        :param env: the environment to wrap
        :param min_perc: minimum percentage of load of the PSN's resources at each reset
        :param max_perc: maximum percentage of load of the PSN's resources at each reset
        :param same_for_all: if True, the same random value is used for all the nodes / links
        """
        super().__init__(env)
        self.same_for_all = same_for_all

        # assert that both min_perc and max_perc are either floats or dicts
        assert (isinstance(min_perc, float) and isinstance(max_perc, float)) or \
               (isinstance(min_perc, dict) and isinstance(max_perc, dict))

        # save the min and max percentages of load
        if isinstance(min_perc, float):
            assert 0 <= min_perc <= 1 and 0 <= max_perc <= 1 and min_perc <= max_perc
            self.min_cpu = self.min_ram = self.min_bw = min_perc
            self.max_cpu = self.max_ram = self.max_bw = max_perc
        else:
            self.min_cpu = min_perc.get('availCPU', 0)
            self.min_ram = min_perc.get('availRAM', 0)
            self.min_bw = min_perc.get('availBW', 0)
            self.max_cpu = max_perc.get('availCPU', 0)
            self.max_ram = max_perc.get('availRAM', 0)
            self.max_bw = max_perc.get('availBW', 0)
            assert 0 <= self.min_cpu <= 1 and 0 <= self.max_cpu <= 1 and self.min_cpu <= self.max_cpu
            assert 0 <= self.min_ram <= 1 and 0 <= self.max_ram <= 1 and self.min_ram <= self.max_ram
            assert 0 <= self.min_bw <= 1 and 0 <= self.max_bw <= 1 and self.min_bw <= self.max_bw

    def reset(self, **kwargs):
        if self.same_for_all:
            self.cpu_load = np.random.uniform(self.min_cpu, self.max_cpu, size=1).item()
            self.ram_load = np.random.uniform(self.min_ram, self.max_ram, size=1).item()
            self.bw_load = np.random.uniform(self.min_bw, self.max_bw, size=1).item()
        return super().reset(**kwargs)
    
    def _init_psn_load(self):
        if self.same_for_all:
            super()._init_psn_load()
        else:
            for _, node in self.env.psn.nodes.items():
                if node['NodeType'] == "server":
                    cpu_load = np.random.uniform(self.min_cpu, self.max_cpu, size=1).item()
                    ram_load = np.random.uniform(self.min_ram, self.max_ram, size=1).item()
                    node['availCPU'] = int(node['CPUcap'] * (1 - cpu_load))
                    node['availRAM'] = int(node['RAMcap'] * (1 - ram_load))
            for _, link in self.env.psn.edges.items():
                bw_load = np.random.uniform(self.min_bw, self.max_bw, size=1).item()
                link['availBW'] = int(link['BWcap'] * (1 - bw_load))
