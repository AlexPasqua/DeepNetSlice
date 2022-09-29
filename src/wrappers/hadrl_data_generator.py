import random
from typing import Tuple, Union, List

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
            n_EDCs: int = 15,
            n_servers_per_DC: Tuple[int, int, int] = (16, 10, 4),
            cpu_cap: int = 50,
            ram_cap: int = 300,
            intra_CCP_bw_cap: int = 100000,  # 100000 Mbps = 100 Gbps
            intra_CDC_bw_cap: int = 100000,  # 100000 Mbps = 100 Gbps
            intra_EDC_bw_cap: int = 10000,  # 10000 Mbps = 10 Gbps
            outer_DC_bw_cap: int = 100000,  # 100000 Mbps = 100 Gbps
    ):
        super().__init__(env)
        self._path = path
        self._create_HADRL_PSN_file(path, n_CCPs, n_CDCs, n_EDCs,
                                    n_servers_per_DC, cpu_cap, ram_cap,
                                    intra_CCP_bw_cap, intra_CDC_bw_cap,
                                    intra_EDC_bw_cap, outer_DC_bw_cap)

    def reset(self, **kwargs):
        # make the env read the PSN file created by this wrapper
        self.unwrapped._psn_file = self._path
        return self.env.reset(**kwargs)

    def _create_HADRL_PSN_file(
            self,
            path: str,
            n_CCPs: int,
            n_CDCs: int,
            n_EDCs: int,
            n_servers_per_DC: Tuple[int, int, int],
            cpu_cap: int,
            ram_cap: int,
            intra_CCP_bw_cap: int,
            intra_CDC_bw_cap: int,
            intra_EDC_bw_cap: int,
            outer_DC_bw_cap: int,
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
        n_routers = n_CCPs + n_CDCs + n_EDCs
        routers_ids = list(range(EDC_ids[-1, -1] + 1,
                                 EDC_ids[-1, -1] + 1 + n_routers))

        # create graph
        g = nx.Graph(Label="HA-DRL PSN")

        # add nodes
        self._create_HADRL_nodes(g, CCP_ids, CDC_ids, EDC_ids, routers_ids,
                                 cpu_cap, ram_cap)

        # add links
        self._create_HADRL_links(
            g, n_CCPs, n_CDCs, n_EDCs, n_servers_per_CCP, n_servers_per_CDC,
            n_servers_per_EDC, CCP_ids, CDC_ids, EDC_ids, routers_ids,
            intra_CCP_bw_cap, intra_CDC_bw_cap, intra_EDC_bw_cap,
            outer_DC_bw_cap)

        # save graph
        nx.write_graphml(g, path)

    @staticmethod
    def _create_HADRL_nodes(
            g: nx.Graph,
            CCP_ids: Union[np.ndarray, List[int]],
            CDC_ids: Union[np.ndarray, List[int]],
            EDC_ids: Union[np.ndarray, List[int]],
            routers_ids: Union[np.ndarray, List[int]],
            cpu_cap: int,
            ram_cap: int,
    ):
        all_server_ids = np.concatenate((CCP_ids.flatten(),
                                         CDC_ids.flatten(),
                                         EDC_ids.flatten()))
        for server_id in all_server_ids:
            g.add_node(server_id, NodeType="server", CPUcap=cpu_cap,
                       RAMcap=ram_cap)
        for router_id in routers_ids:
            g.add_node(router_id, NodeType="router")

    @staticmethod
    def _create_HADRL_links(
            g: nx.Graph,
            n_CCPs: int,
            n_CDCs: int,
            n_EDCs: int,
            n_servers_per_CCP: int,
            n_servers_per_CDC: int,
            n_servers_per_EDC: int,
            CCP_ids: Union[np.ndarray, List[int]],
            CDC_ids: Union[np.ndarray, List[int]],
            EDC_ids: Union[np.ndarray, List[int]],
            routers_ids: Union[np.ndarray, List[int]],
            intra_CCP_bw_cap: int,
            intra_CDC_bw_cap: int,
            intra_EDC_bw_cap: int,
            outer_DC_bw_cap: int,
    ):
        CCPs_routers = routers_ids[:n_CCPs]
        CDCs_routers = routers_ids[n_CCPs:n_CCPs + n_CDCs]
        EDCs_routers = routers_ids[n_CCPs + n_CDCs:]

        # CCPs' servers to their routers
        for i in range(n_CCPs):
            for j in range(n_servers_per_CCP):
                g.add_edge(CCP_ids[i, j], CCPs_routers[i],
                           BWcap=intra_CCP_bw_cap)

        # CDCs' servers to their routers
        for i in range(n_CDCs):
            for j in range(n_servers_per_CDC):
                g.add_edge(CDC_ids[i, j], CDCs_routers[i],
                           BWcap=intra_CDC_bw_cap)

        # EDCs' servers to their routers
        for i in range(n_EDCs):
            for j in range(n_servers_per_EDC):
                g.add_edge(EDC_ids[i, j], EDCs_routers[i],
                           BWcap=intra_EDC_bw_cap)

        # CDCs' routers to CPPs' routers
        for i in range(n_CDCs):
            # each CDC is connected to one CCP
            corresp_CCP = np.random.randint(0, n_CCPs)
            g.add_edge(CDCs_routers[i], CCPs_routers[corresp_CCP],
                       BWcap=outer_DC_bw_cap)

        # connect each CDC's router to n EDCs' routers
        n_EDCs_per_CDC = 3
        for i in range(n_CDCs):
            corresp_EDCs = np.random.choice(n_EDCs, n_EDCs_per_CDC,
                                            replace=False)
            for j in range(n_EDCs_per_CDC):
                g.add_edge(CDCs_routers[i], EDCs_routers[corresp_EDCs[j]],
                           BWcap=outer_DC_bw_cap)

        # connect CDCs and EDCs' routers in a circular way (like in Fig. 1 in HA-DRL paper)
        CDCs_and_EDCs_routers = np.concatenate((CDCs_routers, EDCs_routers))
        for i in range(len(CDCs_and_EDCs_routers)):
            g.add_edge(CDCs_and_EDCs_routers[i],
                       CDCs_and_EDCs_routers[
                           (i + 1) % len(CDCs_and_EDCs_routers)],
                       BWcap=outer_DC_bw_cap)
