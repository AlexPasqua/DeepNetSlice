from typing import Tuple

import gym
import networkx as nx
import numpy as np


class HadrlDataGenerator(gym.Wrapper):
    """
    Wrapper to make the simulator generate data the same way as in the
    paper HA-DRL[1].

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            env: gym.Env,
            path: str,
            n_CCPs: int = 1,
            n_CDCs: int = 5,
            n_EDCs: int = 16,
            n_servers_per_DC: Tuple[int, int, int] = (15, 10, 4),
            cpu_cap: int = 50,
            ram_cap: int = 300,
    ):
        super().__init__(env)
        self._create_HADRL_PSN_file(path, n_CCPs, n_CDCs, n_EDCs,
                                    n_servers_per_DC, cpu_cap, ram_cap)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # TODO: fill here -> overwrite the PSN and the NSPRs
        # self.env.psn = self._init_psn()

        return obs

    def _create_HADRL_PSN_file(
            self,
            path: str,
            n_CCPs: int,
            n_CDCs: int,
            n_EDCs: int,
            n_servers_per_DC: Tuple[int, int, int],
            cpu_cap: int,
            ram_cap: int,
    ):
        """ Initialize the PSN as in the HA-DRL paper

        :param n_CCPs: number of CCPs
        :param n_CDCs: number of CDCs
        :param n_EDCs: number of EDCs
        :param n_servers_per_DC: tuple with the number of servers per (CCP, CDC, EDC)
        """
        # number of servers per DC category
        n_servers_per_CCP, n_servers_per_CDC, n_servers_per_EDC = n_servers_per_DC
        n_ids_CCPs = n_CCPs * n_servers_per_CCP
        n_ids_CDCs = n_CDCs * n_servers_per_CDC
        n_ids_EDCs = n_EDCs * n_servers_per_EDC

        # ids of servers in various DCs
        CCP_ids = np.arange(n_ids_CCPs).reshape(n_CCPs, n_servers_per_CCP)
        CDC_ids = np.arange(
            n_ids_CCPs,
            n_ids_CCPs + n_ids_CDCs).reshape(n_CDCs, n_servers_per_CDC)
        EDC_ids = np.arange(
            CDC_ids[-1, -1] + 1,
            CDC_ids[-1, -1] + 1 + n_ids_EDCs).reshape(n_EDCs, n_servers_per_EDC)

        # one router per DC (based on Fig. 1 in HA-DRL paper)
        # NOTE: the switches are not present in this implementation, but should be equivalent
        n_routers = n_CDCs + n_EDCs
        routers_ids = list(range(EDC_ids[-1, -1] + 1,
                                 EDC_ids[-1, -1] + 1 + n_routers))

        # create graph
        g = nx.Graph(Label="HA-DRL PSN")

        # add nodes
        all_server_ids = np.concatenate((CCP_ids.flatten(),
                                         CDC_ids.flatten(),
                                         EDC_ids.flatten()))
        for server_id in all_server_ids:
            g.add_node(server_id, NodeType="server", CPUcap=cpu_cap, RAMcap=ram_cap)
        for router_id in routers_ids:
            g.add_node(router_id, NodeType="router")

        # add links
        pass
