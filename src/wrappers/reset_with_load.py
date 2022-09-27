from typing import Union

import gym


class ResetWithLoad(gym.Wrapper):
    """ Reset the PSN with a certain amount of load """

    def __init__(self, env: gym.Env, reset_load_perc: Union[float, dict] = 0.):
        """ Constructor

        :param env: :param env: the environment to wrap
        :param reset_load_perc: init percentage of load of the PSN's resources at each reset:
            if float, that value applies to all the resources for all nodes and links;
            if dict, it can specify the load for each type of resource.
        """
        super().__init__(env)
        # define the load percentages of each resource
        if isinstance(reset_load_perc, float):
            self.cpu_load = self.ram_load = self.bw_load = reset_load_perc
        else:
            self.cpu_load = reset_load_perc.get('availCPU', 0)
            self.ram_load = reset_load_perc.get('availRAM', 0)
            self.bw_load = reset_load_perc.get('availBW', 0)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._init_psn_load(self.cpu_load, self.ram_load, self.bw_load)
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
