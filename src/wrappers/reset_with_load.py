import math
import random
from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple

import gym
import networkx as nx
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class ResetWithLoad(gym.Wrapper, ABC):
    """ Abstract class. Wrapper to reset the PSN with a certain tr_load """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.cpu_load = self.ram_load = self.bw_load = 0.

    def reset(self, **kwargs):
        raise NotImplementedError   # doesn't work anymore, needs to be adapted
        self.env.reset(**kwargs)
        self._init_psn_load()
        obs = self.env.update_nspr_state()    # the obs in the env.reset method is outdated
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
    """ Reset the PSN with a certain - fixed - amount of tr_load """

    def __init__(self, env: gym.Env, reset_load_perc: Union[float, dict] = 0.):
        """ Constructor

        :param env: :param env: the environment to wrap
        :param reset_load_perc: init percentage of tr_load of the PSN's resources at each reset:
            if float, that value applies to all the resources for all nodes and links;
            if dict, it can specify the tr_load for each type of resource.
        """
        super().__init__(env)
        assert isinstance(reset_load_perc, (float, dict))
        # define the tr_load percentages of each resource
        if isinstance(reset_load_perc, float):
            assert 0 <= reset_load_perc <= 1
            self.cpu_load = self.ram_load = self.bw_load = reset_load_perc
        else:
            self.cpu_load = reset_load_perc.get('availCPU', 0)
            self.ram_load = reset_load_perc.get('availRAM', 0)
            self.bw_load = reset_load_perc.get('availBW', 0)
            assert 0 <= self.cpu_load <= 1 and 0 <= self.ram_load <= 1 and 0 <= self.bw_load <= 1


class ResetWithRandLoad(ResetWithLoad):
    """ Reset the PSN with a random uniform amount of tr_load """

    def __init__(self, env: gym.Env, min_perc: Union[float, dict],
                 max_perc: Union[float, dict], same_for_all: bool = True):
        """ Constructor

        :param env: the environment to wrap
        :param min_perc: minimum percentage of tr_load of the PSN's resources at each reset
        :param max_perc: maximum percentage of tr_load of the PSN's resources at each reset
        :param same_for_all: if True, the same random value is used for all the nodes / links
        """
        super().__init__(env)
        self.same_for_all = same_for_all

        # assert that both min_perc and max_perc are either floats or dicts
        assert (isinstance(min_perc, float) and isinstance(max_perc, float)) or \
               (isinstance(min_perc, dict) and isinstance(max_perc, dict))

        # save the min and max percentages of tr_load
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


class ResetWithLoadMixed(gym.Wrapper):
    """ Wrapper to reset the PSN with a certain load.
    The load is expressed in percentage and can be resource-specific or general
    (each resource reset with the same load).
    It selects a load percentage for each node/link such that the overall load of
    the PSN is the specified one. It means certain nodes will be free, others
    completely occupied and others will be partially occupied, so that the overall
    CPU/RAM capacity is the specified one. (Same thing for links with their bandwidth).
    """
    def __init__(
            self,
            env: Union[gym.Env, VecEnv],
            load: Union[float, Dict[str, float]] = 0.5,
            rand_load: bool = False,
            rand_range: Tuple[float, float] = (0., 1.)
    ):
        """
        :param env: environment
        :param load: the target load of the PSN, it can be:
            float: single fixed value for all the resources;
            Dict[resource: load]: fixed value but specific for each resource (CPU, RAM, BW)
        :param rand_load: if True, at every 'reset' the PSN's load will be random (same value for all resources);
            note: if 'random' is true, 'load' will be ignored.
        :param rand_range: min and max (included) load values tu consider when 'random' is true
        """
        super(ResetWithLoadMixed, self).__init__(env)
        self.random = rand_load
        self.tot_cpu_cap = self.tot_ram_cap = self.tot_bw_cap = None
        if not rand_load:
            if isinstance(load, float):
                assert 0. <= load <= 1.
                self.cpu_load = self.ram_load = self.bw_load = load
            elif isinstance(load, dict):
                self.cpu_load = load.get('cpu', 0)
                self.ram_load = load.get('ram', 0)
                self.bw_load = load.get('bw', 0)
                assert 0. <= self.cpu_load <= 1. and 0. <= self.ram_load <= 1. and \
                       0. <= self.bw_load <= 1.
            else:
                raise ValueError("Param 'load' is of an incorrect type")
        else:
            assert len(rand_range) == 2 and 0. <= rand_range[0] <= 1. and \
                   0. <= rand_range[1] <= 1.
            min_load, max_load = rand_range
            self.rand_vals = np.arange(min_load, max_load, 0.1)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self._init_psn_load()
        obs = self.env.update_nspr_state()    # the obs in the env.reset method is outdated
        return obs

    def compute_link_weight(self, source, target, link):
        return 1 if link['availBW'] >= self.vl_req_bw else math.inf

    def _init_psn_load(self):
        """ Initialize the PSN's load """
        if self.random:
            load = np.random.choice(self.rand_vals, 1)
            self.cpu_load = self.ram_load = self.bw_load = load[0]

        # TODO: occhio che 'reset' qui viene chiamato da ogni env in VecEnv singolarmente...
        # TODO: quindi, qui, self.env non è VecEnv, ma solo NetworkSimulator
        psns = self.env.get_attr('psn') if isinstance(self.env, VecEnv) else [self.env.psn]
        max_cpus = self.env.get_attr('max_cpu') if isinstance(self.env, VecEnv) else [self.env.max_cpu]
        max_rams = self.env.get_attr('max_ram') if isinstance(self.env, VecEnv) else [self.env.max_ram]
        max_bws = self.env.get_attr('max_bw') if isinstance(self.env, VecEnv) else [self.env.max_bw]
        obs_dicts = self.env.get_attr('obs_dict') if isinstance(self.env, VecEnv) else [self.env.obs_dict]
        maps_id_idx = self.env.get_attr('map_id_idx') if isinstance(self.env, VecEnv) else [self.env.map_id_idx]

        # NOTE: only works if all the envs in the VecEnv use the same PSN
        if self.tot_cpu_cap is None or self.tot_ram_cap is None or self.tot_bw_cap is None:
            self.tot_cpu_cap = self.env.tot_cpu_cap
            self.tot_ram_cap = self.env.tot_ram_cap
            self.tot_bw_cap = self.env.tot_bw_cap

        self.vl_req_bw = 2000
        for i, psn in enumerate(psns):
            max_cpu, max_ram, max_bw = max_cpus[i], max_rams[i], max_bws[i]
            obs_dict, map_id_idx = obs_dicts[i], maps_id_idx[i]
            tot_cpu_to_remove = self.cpu_load * self.tot_cpu_cap / max_cpu
            tot_ram_to_remove = self.ram_load * self.tot_ram_cap / max_ram
            tot_bw_to_remove = self.bw_load * self.tot_bw_cap / max_bw
            # iterate over nodes in a random order and reduce the CPU/RAM availabilities
            nodes = list(psn.nodes.items())
            # random.shuffle(nodes)
            # for node_id, node in nodes:
            n_processed_nodes = 0
            while tot_cpu_to_remove > 0 or tot_ram_to_remove > 0:
                node_id, node = random.sample(nodes, 1)[0]
                if node['NodeType'] == 'server':
                    idx = map_id_idx[node_id]
                    # TODO: consider to extend as [0.25, 0.5, 0.75, 1.]
                    perc_to_remove = np.random.choice([0.5], 1)[0]
                    # CPU to remove
                    # x% of the node capacity (normalized)
                    cur_cpu_to_remove = perc_to_remove * node['CPUcap'] / max_cpu
                    cur_cpu_to_remove = min([round(cur_cpu_to_remove, 3),
                                             tot_cpu_to_remove,
                                             obs_dict['cpu_avails'][idx]])
                    # RAM to remove
                    cur_ram_to_remove = perc_to_remove * node['RAMcap'] / max_ram
                    cur_ram_to_remove = min([round(cur_ram_to_remove, 3),
                                             tot_ram_to_remove,
                                             obs_dict['ram_avails'][idx]])
                    # remove resources
                    obs_dict['cpu_avails'][idx] -= cur_cpu_to_remove
                    obs_dict['ram_avails'][idx] -= cur_ram_to_remove
                    tot_cpu_to_remove -= cur_cpu_to_remove
                    tot_ram_to_remove -= cur_ram_to_remove

                    # from the 2nd node onwards
                    if n_processed_nodes > 0:
                        path = nx.shortest_path(G=psn, source=prev_node_id, target=node_id,
                                                weight=self.compute_link_weight, method='dijkstra')
                        if len(path) >= 2:
                            max_requestable_bw = int(tot_bw_to_remove * max_bw / (2 * len(path) - 2))
                            req_bw = min(self.vl_req_bw, max_requestable_bw)
                            req_bw_normal = round(req_bw / max_bw, 6)
                            for j in range(len(path) - 1):
                                link = psn.edges[path[j], path[j+1]]
                                extr1_idx = map_id_idx[path[j]]
                                extr2_idx = map_id_idx[path[j + 1]]
                                # TODO: req BW by VL = 2000 (hard coded for now)
                                link['availBW'] -= req_bw
                                obs_dict['bw_avails'][extr1_idx] -= req_bw_normal
                                obs_dict['bw_avails'][extr2_idx] -= req_bw_normal
                                tot_bw_to_remove -= req_bw_normal
                                if link['availBW'] < 0 or obs_dict['bw_avails'][extr1_idx] < 0 or obs_dict['bw_avails'][extr2_idx] < 0:
                                    break

                    # save node as previous node
                    prev_node_id = node_id
                    n_processed_nodes += 1

            # # iterate over links in random order and reduce the BW availability
            # links = list(psn.edges.items())
            # # random.shuffle(links)
            # # for extremes, link in links:
            # while tot_bw_to_remove > 0:
            #     extremes, link = random.sample(links, 1)[0]
            #     cur_bw_to_remove = np.random.randint(0, link['availBW'], 1)[0]
            #     cur_bw_to_remove = min(cur_bw_to_remove, tot_bw_to_remove * max_bw)
            #     cur_bw_to_remove_normal = cur_bw_to_remove / max_bw
            #     idx_0, idx_1 = map_id_idx[extremes[0]], map_id_idx[extremes[1]]
            #     # links' BW actually reduced because needed for shortest path calculation
            #     link['availBW'] -= cur_bw_to_remove
            #     obs_dict['bw_avails'][idx_0] -= cur_bw_to_remove_normal
            #     obs_dict['bw_avails'][idx_1] -= cur_bw_to_remove_normal
            #     tot_bw_to_remove -= cur_bw_to_remove_normal

        return


class ResetWithLoadBinary(ResetWithLoadMixed):
    """ Wrapper to reset the PSN with a certain load.
        The load is expressed in percentage and can be resource-specific or general
        (each resource reset with the same load).
        It put a certain amount of nodes with zero available resources, so that
        the overall load of the PSN is the one specified.

        Note: only the CPU and RAM are modified, not the bandwidth
        """

    def __init__(
            self,
            env: Union[gym.Env, VecEnv],
            load: Union[float, Dict[str, float]] = 0.5,
            rand_load: bool = False,
            rand_range: Tuple[float, float] = (0., 1.)
    ):
        """
        :param env: environment
        :param load: the target load of the PSN, it can be:
            float: single fixed value for all the resources;
            Dict[resource: load]: fixed value but specific for each resource (CPU, RAM, BW)
        :param rand_load: if True, at every 'reset' the PSN's load will be random (same value for all resources);
            note: if 'random' is true, 'load' will be ignored.
        :param rand_range: min and max (included) load values tu consider when 'random' is true
        """
        super().__init__(env, load, rand_load, rand_range)

    def _init_psn_load(self):
        """ Initialize the PSN's load """
        if self.random:
            load = np.random.choice(self.rand_vals, 1)
            self.cpu_load = self.ram_load = self.bw_load = load[0]

        psns = self.env.get_attr('psn') if isinstance(self.env, VecEnv) else [self.env.psn]
        max_cpus = self.env.get_attr('max_cpu') if isinstance(self.env, VecEnv) else [self.env.max_cpu]
        max_rams = self.env.get_attr('max_ram') if isinstance(self.env, VecEnv) else [self.env.max_ram]
        max_bws = self.env.get_attr('max_bw') if isinstance(self.env, VecEnv) else [self.env.max_bw]
        obs_dicts = self.env.get_attr('obs_dict') if isinstance(self.env, VecEnv) else [self.env.obs_dict]
        maps_id_idx = self.env.get_attr('map_id_idx') if isinstance(self.env, VecEnv) else [self.env.map_id_idx]

        if self.tot_cpu_cap is None or self.tot_ram_cap is None or self.tot_bw_cap is None:
            self.tot_cpu_cap = self.env.tot_cpu_cap
            self.tot_ram_cap = self.env.tot_ram_cap
            self.tot_bw_cap = self.env.tot_bw_cap

        for i, psn in enumerate(psns):
            max_cpu, max_ram, max_bw = max_cpus[i], max_rams[i], max_bws[i]
            obs_dict, map_id_idx = obs_dicts[i], maps_id_idx[i]
            tot_cpu_to_remove = self.cpu_load * self.tot_cpu_cap / max_cpu
            tot_ram_to_remove = self.ram_load * self.tot_ram_cap / max_ram
            tot_bw_to_remove = self.bw_load * self.tot_bw_cap / max_bw
            # iterate over nodes in a random order and reduce the CPU/RAM availabilities
            nodes = list(psn.nodes.items())
            while tot_cpu_to_remove > 0 or tot_ram_to_remove > 0:
                node_id, node = random.sample(nodes, 1)[0]
                if node['NodeType'] == 'server':
                    idx = map_id_idx[node_id]
                    cur_removed_cpu = obs_dict['cpu_avails'][idx]
                    obs_dict['cpu_avails'][idx] = 0.
                    obs_dict['ram_avails'][idx] = 0.
                    tot_cpu_to_remove -= cur_removed_cpu
                    tot_ram_to_remove -= cur_removed_cpu


class ResetWithRealisticLoad(gym.Wrapper):
    """ Wrapper that resets the PSN with a certain amount of load already.
    It does so in a way that resembles a how the state of the PSN might be in
    case an agent has been actually placing NSPRs.

    It samples NSPRs from the ones that should arrive during the current episode
    and place their VNFs in random nodes and connects them via shortest path.
    This way the CPU/RAM and even the BW allocation should be realistic.
    """

    def __init__(self, env: gym.Env, cpu_load: float, **kwargs):
        """
        :param env: environment
        :param cpu_load: target percentage of CPU load of the PSN
        """
        super().__init__(env)
        assert 0. <= cpu_load <= 1.
        self.cpu_load = cpu_load

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.init_psn_load()
        obs = self.env.update_nspr_state()  # the obs in the env.reset method is outdated
        return obs

    def init_psn_load(self):
        """ Initialize the PSN with the target load """
        cpu_to_remove_normal = self.env.tot_cpu_cap * self.cpu_load / self.env.max_cpu
        removed_cpu_normal = 0
        while removed_cpu_normal < cpu_to_remove_normal:
            nspr = self.sample_nspr()
            placement_map = {}
            # place all VNFs
            for vnf_id, vnf in nspr.nodes.items():
                node_id, node_idx = self.sample_suitable_node(vnf)
                placement_map[vnf_id] = node_id
                self.env.obs_dict['cpu_avails'][node_idx] -= vnf['reqCPU'] / self.env.max_cpu
                self.env.obs_dict['ram_avails'][node_idx] -= vnf['reqRAM'] / self.env.max_ram
                removed_cpu_normal += vnf['reqCPU'] / self.env.max_cpu
                if removed_cpu_normal >= cpu_to_remove_normal:
                    break
            # place all VLs
            for (src_vnf_id, dst_vnf_id), vl in nspr.edges.items():
                self.req_bw = vl['reqBW']
                try:
                    src_node_id = placement_map[src_vnf_id]
                    dst_node_id = placement_map[dst_vnf_id]
                except KeyError:
                    # it means either src_vnf_id, dst_vnf_id or both hasn't been placed -> skip link placement
                    continue
                path = nx.shortest_path(G=self.env.psn, source=src_node_id,
                                        target=dst_node_id, weight=self.compute_links_weights,
                                        method='dijkstra')
                for i in range(len(path) - 1):
                    self.env.psn.edges[path[i], path[i+1]]['availBW'] -= vl['reqBW']
                    idx1 = self.env.map_id_idx[path[i]]
                    idx2 = self.env.map_id_idx[path[i+1]]
                    self.env.obs_dict['bw_avails'][idx1] -= vl['reqBW'] / self.env.max_bw
                    self.env.obs_dict['bw_avails'][idx2] -= vl['reqBW'] / self.env.max_bw

    def compute_links_weights(self, source, target, link):
        """ Method called automatically by nx.shortest_path() """
        return 1 if link['availBW'] >= self.req_bw else math.inf

    def sample_suitable_node(self, vnf: dict):
        """ Sample a random node with enough resources to host the VNF """
        server_idx = np.random.choice(list(self.env.servers_map_idx_id.keys()))
        server_id = self.env.servers_map_idx_id[server_idx]
        while not self.env.enough_avail_resources(server_id, vnf):
            server_idx = np.random.choice(list(self.env.servers_map_idx_id.keys()))
            server_id = self.env.servers_map_idx_id[server_idx]
        return server_id, server_idx

    def sample_nspr(self):
        """ Sample a NSPR among the ones that will arrive in this episode """
        arr_time = np.random.choice(list(self.env.nsprs.keys()))
        idx = np.random.choice(len(self.env.nsprs[arr_time]))
        nspr = self.env.nsprs[arr_time][idx]
        return nspr
