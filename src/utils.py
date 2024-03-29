from typing import Tuple, Union, List, Optional, Type

import gym
import networkx as nx
import numpy as np

from gym.utils.env_checker import check_env
from network_simulator import NetworkSimulator
from wrappers import NSPRsGeneratorHADRL, RemovePlacementState, DynamicConnectivity
from sb3_contrib.common.wrappers import ActionMasker


def make_env(
        psn_path: str,
        base_env_kwargs: Optional[dict] = None,
        time_limit: bool = False,
        time_limit_kwargs: Optional[dict] = None,
        reset_load_class: Type[gym.Wrapper] = None,
        reset_load_kwargs: Optional[dict] = None,
        generate_nsprs: bool = False,
        nsprs_gen_kwargs: Optional[dict] = None,
        placement_state: bool = True,
        dynamic_connectivity: bool = False,
        dynamic_connectivity_kwargs: Optional[dict] = dict(link_bw=10_000),
        dynamic_topology: bool = False,
):
    """ Create the environment.
    It can be wrapped with different wrappers, all with their own arguments.
    They wrappers are namely: TimeLimit, ResetWithRandLoad, NSPRsGeneratorHADRL.

    :param psn_path: path to the PSN file
    :param base_env_kwargs: kwargs of the base environment
    :param time_limit: if True, the env is wrapped with TimeLimit wrapper
    :param time_limit_kwargs: kwargs of the TimeLimit wrapper
    :param reset_load_class: class of the wrapper to reset the PSN with load
    :param reset_load_kwargs: kwargs for the reset-with-load wrapper
    :param hadrl_nsprs: if True, the env is wrapped with NSPRsGeneratorHADRL wrapper
    :param hadrl_nsprs_kwargs: kwargs for the NSPRsGeneratorHADRL wrapper
    :param placement_state: if False, adds a wrapper that removes the placement state from the observations
    :param dynamic_connectivity: if True, the connectivity of the PSN changes in every episode
    :param dynamic_connectivity_kwargs: kwargs for the DynamicConnectivity wrapper
    :param dynamic_topology: if True, the topology of the PSN changes in every episode.
        Note: it True, 'dynamic_connectivity' will be forced to True as well,
        as there's no way to change the nodes and not the connectivity.
    """
    base_env_kwargs = {} if base_env_kwargs is None else base_env_kwargs
    time_limit_kwargs = {} if time_limit_kwargs is None else time_limit_kwargs
    reset_load_kwargs = {} if reset_load_kwargs is None else reset_load_kwargs
    dynamic_connectivity_kwargs = {} if dynamic_connectivity_kwargs is None else dynamic_connectivity_kwargs

    # base env
    env = NetworkSimulator(psn_path, **base_env_kwargs)

    # apply wrappers
    if time_limit:
        env = gym.wrappers.TimeLimit(env, **time_limit_kwargs)
    if generate_nsprs:
        env = NSPRsGeneratorHADRL(env, **nsprs_gen_kwargs)
    if dynamic_topology:
        env = ActionMasker(env, action_mask_fn=env.get_action_mask)
        env = DynamicConnectivity(env, nodes_mask=env.get_action_mask, **dynamic_connectivity_kwargs)
        dynamic_connectivity = False
    if dynamic_connectivity:
        env = DynamicConnectivity(env, **dynamic_connectivity_kwargs)
    if reset_load_class is not None:
        env = reset_load_class(env, **reset_load_kwargs)
    if not placement_state:
        env = RemovePlacementState(env)
    # check_env(env)  # could make the code crash with masked actions
    return env


def create_HADRL_PSN_file(
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
        n_EDCs_per_CDC: int = 3,
):
    """ Initialize the PSN as in the HA-DRL paper

    :param path: path where to save the file defining the PSN
    :param n_CCPs: number of CCPs
    :param n_CDCs: number of CDCs
    :param n_EDCs: number of EDCs
    :param n_servers_per_DC: tuple with the number of servers per (CCP, CDC, EDC)
    :param cpu_cap: CPU capacity per server
    :param ram_cap: RAM capacity per server
    :param intra_CCP_bw_cap: bandwidth of links within a CCP
    :param intra_CDC_bw_cap: bandwidth of links within a CDC
    :param intra_EDC_bw_cap: bandwidth of links within a EDC
    :param outer_DC_bw_cap: bandwidth of links between DCs
    :param n_EDCs_per_CDC: number of EDCs connected to each CDC
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

    # one switch per DC (based on Fig. 1 in HA-DRL paper)
    n_switches = n_CCPs + n_CDCs + n_EDCs
    switches_ids = list(range(EDC_ids[-1, -1] + 1,
                              EDC_ids[-1, -1] + 1 + n_switches))

    # one router per DC (based on Fig. 1 in HA-DRL paper)
    n_routers = n_CCPs + n_CDCs + n_EDCs
    routers_ids = list(range(switches_ids[-1] + 1, switches_ids[-1] + 1 + n_routers))

    # create graph
    g = nx.Graph(Label="HA-DRL PSN")

    # add nodes
    _create_nodes(g, CCP_ids, CDC_ids, EDC_ids, switches_ids, routers_ids,
                  cpu_cap, ram_cap)

    # add links
    _create_HADRL_links(
        g, n_CCPs, n_CDCs, n_EDCs, n_servers_per_CCP, n_servers_per_CDC,
        n_servers_per_EDC, CCP_ids, CDC_ids, EDC_ids, switches_ids, routers_ids,
        intra_CCP_bw_cap, intra_CDC_bw_cap, intra_EDC_bw_cap, outer_DC_bw_cap,
        n_EDCs_per_CDC)

    # save graph
    nx.write_graphml(g, path)


def create_HEENSO_PSN_file(
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
        n_EDCs_per_CDC: int = 3,
):
    """ Initialize the PSN as in the paper "Heuristic for Edge-enable Network Slice Optimization
    using the Power of Two Choices"

    Disclaimer: the topology is slightly different, the ring of nodes in Fig. 4
    of the paper is brought one step closer to the CCP and nodes 26 to 30 are
    removed, since they don't increase the number of possible paths across the PSN
    (they would only make some paths 1 step longer, reducing the reward).

    :param path: path where to save the file defining the PSN
    :param n_CCPs: number of CCPs
    :param n_CDCs: number of CDCs
    :param n_EDCs: number of EDCs
    :param n_servers_per_DC: tuple with the number of servers per (CCP, CDC, EDC)
    :param cpu_cap: CPU capacity per server
    :param ram_cap: RAM capacity per server
    :param intra_CCP_bw_cap: bandwidth of links within a CCP
    :param intra_CDC_bw_cap: bandwidth of links within a CDC
    :param intra_EDC_bw_cap: bandwidth of links within a EDC
    :param outer_DC_bw_cap: bandwidth of links between DCs
    :param n_EDCs_per_CDC: number of EDCs connected to each CDC
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

    # one switch per DC (based on Fig. 4 in HEENSO paper)
    n_switches = n_CCPs + n_CDCs + n_EDCs
    switches_ids = list(range(EDC_ids[-1, -1] + 1,
                              EDC_ids[-1, -1] + 1 + n_switches))

    # one router per DC (based on Fig. 4 in HEENSO paper)
    n_routers = n_CDCs + n_EDCs
    routers_ids = list(
        range(switches_ids[-1] + 1, switches_ids[-1] + 1 + n_routers))

    # create graph
    g = nx.Graph(Label="HEENSO PSN")

    # add nodes
    _create_nodes(g, CCP_ids, CDC_ids, EDC_ids, switches_ids, routers_ids,
                  cpu_cap, ram_cap)

    # add links
    _create_HEENSO_links(
        g, n_CCPs, n_CDCs, n_EDCs, n_servers_per_CCP, n_servers_per_CDC,
        n_servers_per_EDC, CCP_ids, CDC_ids, EDC_ids, switches_ids, routers_ids,
        intra_CCP_bw_cap, intra_CDC_bw_cap, intra_EDC_bw_cap, outer_DC_bw_cap,
        n_EDCs_per_CDC)

    # save graph
    nx.write_graphml(g, path)


def _create_nodes(
        g: nx.Graph,
        CCP_ids: Union[np.ndarray, List[int]],
        CDC_ids: Union[np.ndarray, List[int]],
        EDC_ids: Union[np.ndarray, List[int]],
        switches_ids: Union[np.ndarray, List[int]],
        routers_ids: Union[np.ndarray, List[int]],
        cpu_cap: int,
        ram_cap: int,
):
    all_server_ids = np.concatenate((CCP_ids.flatten(),
                                     CDC_ids.flatten(),
                                     EDC_ids.flatten()))
    for server_id in all_server_ids:
        g.add_node(server_id, NodeType="server", CPUcap=cpu_cap, RAMcap=ram_cap)
    for switch_id in switches_ids:
        g.add_node(switch_id, NodeType="switch")
    for router_id in routers_ids:
        g.add_node(router_id, NodeType="router")


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
        switches_ids: Union[np.ndarray, List[int]],
        routers_ids: Union[np.ndarray, List[int]],
        intra_CCP_bw_cap: int,
        intra_CDC_bw_cap: int,
        intra_EDC_bw_cap: int,
        outer_DC_bw_cap: int,
        n_EDCs_per_CDC: int
):
    connect_CDCs_EDCs_randomly = False if n_EDCs / n_CDCs == n_EDCs_per_CDC else True
    CCPs_switches = switches_ids[:n_CCPs]
    CDCs_switches = switches_ids[n_CCPs:n_CCPs + n_CDCs]
    EDCs_switches = switches_ids[n_CCPs + n_CDCs:]
    CCPs_routers = routers_ids[:n_CCPs]
    CDCs_routers = routers_ids[n_CCPs:n_CCPs + n_CDCs]
    EDCs_routers = routers_ids[n_CCPs + n_CDCs:]

    # connect CCPs' servers to their switches
    for i in range(n_CCPs):
        for j in range(n_servers_per_CCP):
            g.add_edge(CCP_ids[i, j], CCPs_switches[i], BWcap=intra_CCP_bw_cap)

    # connect CDCs' servers to their switches
    for i in range(n_CDCs):
        for j in range(n_servers_per_CDC):
            g.add_edge(CDC_ids[i, j], CDCs_switches[i], BWcap=intra_CDC_bw_cap)

    # connect EDCs' servers to their switches
    for i in range(n_EDCs):
        for j in range(n_servers_per_EDC):
            g.add_edge(EDC_ids[i, j], EDCs_switches[i], BWcap=intra_EDC_bw_cap)

    # connect CCPs' switches to their routers
    for i in range(len(CCPs_switches)):
        g.add_edge(CCPs_switches[i], CCPs_routers[i], BWcap=intra_CCP_bw_cap)

    # connect CDCs' switches to their routers
    for i in range(len(CDCs_switches)):
        g.add_edge(CDCs_switches[i], CDCs_routers[i], BWcap=intra_CDC_bw_cap)

    # connect EDCs' switches to their routers
    for i in range(len(EDCs_switches)):
        g.add_edge(EDCs_switches[i], EDCs_routers[i], BWcap=intra_EDC_bw_cap)

    # connect CDCs' routers to CCPs' routers
    for i in range(n_CDCs):
        # each CDC is connected to one CCP
        corresp_CCP = np.random.randint(0, n_CCPs)
        g.add_edge(CDCs_routers[i], CCPs_routers[corresp_CCP], BWcap=outer_DC_bw_cap)

    # connect each CDCs' router to n EDCs' routers
    for i in range(n_CDCs):
        if connect_CDCs_EDCs_randomly:
            corresp_EDCs = np.random.choice(n_EDCs, n_EDCs_per_CDC, replace=False)
        else:
            corresp_EDCs = list(range(n_EDCs_per_CDC * i, n_EDCs * i + n_EDCs_per_CDC))
        for j in range(n_EDCs_per_CDC):
            g.add_edge(CDCs_routers[i], EDCs_routers[corresp_EDCs[j]],
                       BWcap=outer_DC_bw_cap)

    # connect CDCs and EDCs' routers in a circular way (like in Fig. 1 in HA-DRL paper)
    CDCs_and_EDCs_routers = np.concatenate((CDCs_routers, EDCs_routers))
    for i in range(len(CDCs_and_EDCs_routers)):
        g.add_edge(CDCs_and_EDCs_routers[i],
                   CDCs_and_EDCs_routers[(i + 1) % len(CDCs_and_EDCs_routers)],
                   BWcap=outer_DC_bw_cap)


def _create_HEENSO_links(
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
        switches_ids: Union[np.ndarray, List[int]],
        routers_ids: Union[np.ndarray, List[int]],
        intra_CCP_bw_cap: int,
        intra_CDC_bw_cap: int,
        intra_EDC_bw_cap: int,
        outer_DC_bw_cap: int,
        n_EDCs_per_CDC: int
):
    connect_CDCs_EDCs_randomly = False if n_EDCs / n_CDCs == n_EDCs_per_CDC else True
    CCPs_switches = switches_ids[:n_CCPs]
    CDCs_switches = switches_ids[n_CCPs:n_CCPs + n_CDCs]
    EDCs_switches = switches_ids[n_CCPs + n_CDCs:]
    CDCs_routers = routers_ids[:n_CDCs]
    EDCs_routers = routers_ids[n_CDCs:]

    # connect CCPs' servers to their switches
    for i in range(n_CCPs):
        for j in range(n_servers_per_CCP):
            g.add_edge(CCP_ids[i, j], CCPs_switches[i], BWcap=intra_CCP_bw_cap)

    # connect CDCs' servers to their switches
    for i in range(n_CDCs):
        for j in range(n_servers_per_CDC):
            g.add_edge(CDC_ids[i, j], CDCs_switches[i], BWcap=intra_CDC_bw_cap)

    # connect EDCs' servers to their switches
    for i in range(n_EDCs):
        for j in range(n_servers_per_EDC):
            g.add_edge(EDC_ids[i, j], EDCs_switches[i], BWcap=intra_EDC_bw_cap)

    # connect CDCs' switches to their routers
    for i in range(len(CDCs_switches)):
        g.add_edge(CDCs_switches[i], CDCs_routers[i], BWcap=intra_CDC_bw_cap)

    # connect EDCs' switches to their routers
    for i in range(len(EDCs_switches)):
        g.add_edge(EDCs_switches[i], EDCs_routers[i], BWcap=intra_EDC_bw_cap)

    # connect CDCs' routers to CCPs' switches
    for i in range(n_CDCs):
        # each CDC is connected to one CCP
        corresp_CCP = np.random.randint(0, n_CCPs)
        g.add_edge(CDCs_routers[i], CCPs_switches[corresp_CCP], BWcap=outer_DC_bw_cap)

    # connect each CDCs' switch to n EDCs' routers
    for i in range(n_CDCs):
        if connect_CDCs_EDCs_randomly:
            corresp_EDCs = np.random.choice(n_EDCs, n_EDCs_per_CDC, replace=False)
        else:
            corresp_EDCs = list(range(n_EDCs_per_CDC * i, n_EDCs_per_CDC * i + n_EDCs_per_CDC))
        for j in range(n_EDCs_per_CDC):
            g.add_edge(CDCs_switches[i], EDCs_routers[corresp_EDCs[j]],
                       BWcap=outer_DC_bw_cap)

    # connect CDCs routers in a circular way (like in Fig. 4 in HEENSO paper)
    for i in range(len(CDCs_routers)):
        g.add_edge(CDCs_routers[i],
                   CDCs_routers[(i + 1) % len(CDCs_routers)],
                   BWcap=outer_DC_bw_cap)
