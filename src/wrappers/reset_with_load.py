from typing import Union

import gym


class ResetWithLoad(gym.Wrapper):
    """Reset the environment with a load."""
    def __init__(self, env: gym.Env, reset_load_perc: Union[float, dict] = 0.):
        super().__init__(env)
        self.load = reset_load_perc

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # initialize the PSN's load status
        if isinstance(self.reset_load_perc, float):
            cpu_load = ram_load = bw_load = self.reset_load_perc
        else:
            cpu_load = self.reset_load_perc.get('availCPU', 0)
            ram_load = self.reset_load_perc.get('availRAM', 0)
            bw_load = self.reset_load_perc.get('availBW', 0)
        self._init_psn_load(cpu_load, ram_load, bw_load)
        return obs

    def _init_psn_load(self, cpu_load_perc: float, ram_load_perc: float,
                       bw_load_perc: float):
        """ Initialize the PSN's load with the specified values

        :param cpu_load_perc: the percentage of CPU load for each node
        :param ram_load_perc: the percentage of RAM load for each node
        :param bw_load_perc: the percentage of bandwidth load for each link
        """
        for _, node in self.env.psn.nodes.items():
            if node['NodeType'] == "server":
                node['availCPU'] = int(node['CPUcap'] * (1 - cpu_load_perc))
                node['availRAM'] = int(node['RAMcap'] * (1 - ram_load_perc))
        for _, link in self.env.psn.edges.items():
            link['availBW'] = int(link['BWcap'] * (1 - bw_load_perc))
