import copy
import math
from typing import Optional, Union, Tuple

import gym
import networkx as nx
import numpy as np

from gym.spaces import Dict, Box, Discrete

import reader

GymObs = Union[Tuple, dict, np.ndarray, int]


class NetworkSimulator(gym.Env):
    """ Class implementing the network simulator (RL environment) """

    def __init__(
            self,
            psn_file: str,
            nsprs_path: str = "../NSPRs/",
            nsprs_per_episode: int = None,
            nsprs_max_duration: int = 100,
            accumulate_reward: bool = True,
            discount_acc_rew: bool = True,
            perc_avail_nodes: Optional[float] = 1.
    ):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        :param nsprs_per_episode: max number of NSPRs to be evaluated in each episode. If None, there is no limit.
        :param nsprs_max_duration: (optional) max duration of the NSPRs.
        :param accumulate_reward: if true, the reward is accumulated and given to the agent only after each NSPRs
        :param discount_acc_rew: if true, an increasing discount factor is applied to the acceptance reward during each NSPR.
            It starts from the inverse of the number of VNFs in the NSPR and grows to 1.
        :param perc_avail_nodes: in case some action masking is implemented (i.e., env wrapped in ActionMasker
            wrapper from sbe-contrib), it specifies the percentage of available nodes we.r.t. the total.
        """
        super(NetworkSimulator, self).__init__()

        self.psn_file = psn_file
        self.psn = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs_path = nsprs_path
        self.nsprs_per_episode = nsprs_per_episode
        self.accumulate_reward = accumulate_reward
        self.nsprs_seen_in_cur_ep = 0
        self.nsprs_max_duration = nsprs_max_duration
        self.done = False
        self.nsprs = None  # will be initialized in the reset method
        self.waiting_nsprs = []  # list of NSPRs that arrived already and are waiting to be evaluated
        self.cur_nspr = None  # used to keep track of the current NSPR being evaluated
        self.cur_nspr_unplaced_vnfs_ids = []  # used to keep track of the VNFs' IDs of the current NSPR that haven't been placed yet
        self.cur_vnf_id = None  # used to keep track of the current VNF being evaluated
        self._cur_vl_reqBW = 0  # auxiliary attribute needed in method 'self.compute_link_weight'
        self.time_step = 0  # keep track of current time step
        self.ep_number = 0  # keep track of current episode number
        self.tot_seen_nsprs = 0  # keep track of the number of NSPRs seen so far
        self.accepted_nsprs = 0  # for the overall acceptance ratio
        self.discount_acc_rew = discount_acc_rew    # whether or not to discount the acceptance reward
        self.acc_rew_disc_fact = 1.     # current discount factor for the acceptance reward
        self.base_acc_rew_disc_fact = 1.    # base discount factor for the acceptance reward

        # map (dict) between IDs of PSN's nodes and their respective index (see self._init_map_id_idx's docstring)
        nodes_ids = list(self.psn.nodes.keys())
        self.map_id_idx = {nodes_ids[idx]: idx for idx in range(len(nodes_ids))}

        # map (dict) between an index of a list (incrementing int) and the ID of a server
        servers_ids = [node_id for node_id, node in self.psn.nodes.items()
                       if node['NodeType'] == 'server']
        self.servers_map_idx_id = {idx: servers_ids[idx] for idx in range(len(servers_ids))}

        # partial rewards to be accumulated across the steps of evaluation of a single NSPR
        self._acceptance_rewards = []
        self._resource_consumption_rewards = []
        self._cur_resource_consumption_rewards = []
        self._load_balance_rewards = []

        # reward values for specific outcomes
        self.rval_accepted_vnf = 100
        self.rval_rejected_vnf = -100

        # Action space and observation space (gym.Env required attributes)
        ONE_BILLION = 1_000_000_000  # constant for readability
        n_nodes = len(self.psn.nodes)
        # action space = number of servers
        self.action_space = Discrete(len(servers_ids))
        self.observation_space = Dict({
            # PSN STATE
            'cpu_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),
            'ram_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),
            # for each physical node, sum of the BW of the physical links connected to it
            'bw_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),
            # for each physical node, number of VNFs of the current NSPR placed on it
            'placement_state': Box(low=0, high=ONE_BILLION, shape=(n_nodes,), dtype=int),

            # NSPR STATE
            # note: apparently it's not possible to pass "math.inf" or "sys.maxsize" as a gym.spaces.Box's high value
            'cur_vnf_cpu_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            'cur_vnf_ram_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            # sum of the required BW of each VL connected to the current VNF
            'cur_vnf_bw_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            'vnfs_still_to_place': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
        })
        self._empty_psn_obs_dict = None     # used to store the observation resulting from an empty PSN
        self.obs_dict = self._init_obs_dict()     # used to store the current observation

        # action mask determining available actions. Init with all actions are available (it will be update in 'reset')
        self._action_mask = np.ones(shape=(len(servers_ids),), dtype=bool)
        assert 0. <= perc_avail_nodes <= 1.
        self.perc_avail_nodes = perc_avail_nodes

    @property
    def cur_vnf(self):
        return self.cur_nspr.nodes[self.cur_vnf_id] if self.cur_nspr is not None else None
    
    def get_action_mask(self, env):
        # 'action_mask' needs to be callable to be passed ActionMasker wrapper
        # note: env needs to be an argument for compatibility, but in this case it's useless
        return self._action_mask
    
    def reset_partial_rewards(self):
        """ Resets the partial rewards (used in case a NSPR cannot be placed) """
        self._acceptance_rewards = []
        self._resource_consumption_rewards = []
        self._load_balance_rewards = []

    def enough_avail_resources(self, physical_node_id: int, vnf: dict) -> bool:
        """ Check that the physical node has enough resources to satisfy the VNF's requirements

        :param physical_node_id: ID of the physical node to check
        :param vnf: VNF to check
        :return: True if the physical node has enough resources to satisfy the VNF's requirements, False otherwise
        """
        idx = self.map_id_idx[physical_node_id]
        enough_cpu = self.obs_dict['cpu_avails'][idx] >= vnf['reqCPU'] / self.max_cpu
        enough_ram = self.obs_dict['ram_avails'][idx] >= vnf['reqRAM'] / self.max_ram
        return enough_cpu and enough_ram

    def restore_avail_resources(self, nspr: nx.Graph):
        """ Method called in case a NSPR is not accepted, or it has reached
        its departure time.
        Restores the PSN resources occupied by that NSPR.

        :param nspr: the rejected NSPR
        """
        if nspr is not None:
            nspr.graph['departed'] = True
            for vnf_id, vnf in nspr.nodes.items():
                # restore nodes' resources availabilities
                if vnf['placed'] >= 0:
                    idx = self.map_id_idx[vnf['placed']]
                    self.obs_dict['cpu_avails'][idx] += vnf['reqCPU'] / self.max_cpu
                    self.obs_dict['ram_avails'][idx] += vnf['reqRAM'] / self.max_ram
                    self.obs_dict['placement_state'][idx] -= 1
            for _, vl in nspr.edges.items():
                # restore links' resources availabilities
                if vl['placed']:
                    # vl['placed'] is the list of the physical nodes traversed by the link
                    rewBW_normalized = vl['reqBW'] / self.max_bw
                    for i in range(len(vl['placed']) - 1):
                        id_1 = vl['placed'][i]
                        id_2 = vl['placed'][i + 1]
                        physical_link = self.psn.edges[id_1, id_2]
                        # recall that BW in physical links is actually updated
                        physical_link['availBW'] += vl['reqBW']
                        idx_1 = self.map_id_idx[id_1]
                        idx_2 = self.map_id_idx[id_2]
                        self.obs_dict['bw_avails'][idx_1] += rewBW_normalized
                        self.obs_dict['bw_avails'][idx_2] += rewBW_normalized

    def pick_next_nspr(self):
        """ Pick the next NSPR to be evaluated and updates the attribute 'self.cur_nspr' """
        if self.cur_nspr is None and self.waiting_nsprs:
            self.cur_nspr = self.waiting_nsprs.pop(0)
            self.cur_nspr.graph['DepartureTime'] = self.time_step + self.cur_nspr.graph['duration']
            self.cur_nspr_unplaced_vnfs_ids = list(self.cur_nspr.nodes.keys())
            self.cur_vnf_id = self.cur_nspr_unplaced_vnfs_ids.pop(0)
            # reset acceptance reward discount factor
            self.base_acc_rew_disc_fact = 1 / len(self.cur_nspr.nodes)
            self.acc_rew_disc_fact = 0.
            # self.tot_seen_nsprs += 1
            _ = self.update_nspr_state()    # obs_dict updated within method

    def check_for_departed_nsprs(self):
        """ Checks it some NSPRs have reached their departure time and in case
        it frees the PSN resources occupied by them. """
        all_arrival_times = list(self.nsprs.keys())
        all_arrival_times.sort()
        for arrival_time in all_arrival_times:
            if arrival_time >= self.time_step:
                break
            cur_nsprs = self.nsprs[arrival_time]
            for nspr in cur_nsprs:
                departed = nspr.graph.get('departed', False)
                if nspr.graph.get('DepartureTime', self.time_step) < self.time_step and not departed:
                    self.restore_avail_resources(nspr=nspr)

                    # This should be useless now
                    # if nspr == self.cur_nspr:
                    #     # haven't finished placing this NSPR, but its departure time has come.
                    #     # remove NSPR, no reward, neither positive nor negative
                    #     # (not agent's fault, too many requests at the same time)
                    #     self.cur_nspr = None
                    #     self.reset_partial_rewards()

    def manage_unsuccessful_action(self) -> Tuple[GymObs, int]:
        """ Method to manage an unsuccessful action, executed when a VNF/VL cannot be placed onto the PSN.
        - Restore the PSN resources occupied by VNFs and VLs of the current NSPR
        - Reset the partial rewards
        - Set the reward as the one for an unsuccessful action
        - Pick the next NSPR to be evaluated (if exists)
        - get an observation from the environment

        :return: the reward for the unsuccessful action
        """
        self.restore_avail_resources(nspr=self.cur_nspr)
        self.reset_partial_rewards()
        self.cur_nspr = None
        self.nsprs_seen_in_cur_ep += 1

        self.tot_seen_nsprs += 1
        if self.nsprs_seen_in_cur_ep >= self.nsprs_per_episode:
            self.done = True
        self.waiting_nsprs += self.nsprs.get(self.time_step, [])
        self.pick_next_nspr()
        obs = self.update_nspr_state()
        reward = self.rval_rejected_vnf
        self.time_step += 1
        return obs, reward

    def _normalize_reward_0_10(self, reward):
        """ Normalize the reward to be in [0, 10] (as in HA-DRL) """
        # since the global reward is given by the sum for each time step of the
        # current NSPR (i.e. for each VNF in the NSPR) of the product of the 3
        # partial rewards at time t,
        # the maximum possible reward for the given NSPR is given by:
        #   the number of VNF in the NSPR times
        #   the maximum acceptance reward value (i.e. every VNF is accepted) times
        #   the maximum resource consumption reward value (i.e. 1) times
        #   the maximum tr_load balancing reward value (i.e. 1+1=2)
        max_reward = len(self.cur_nspr.nodes) * self.rval_accepted_vnf * 1 * 2
        return reward / max_reward * 10

    @staticmethod
    def get_cur_vnf_vls(vnf_id: int, nspr: nx.Graph) -> dict:
        """ Get all the virtual links connected to a specific VNF

        :param vnf_id: ID of a VNF whose VLs have to be returned
        :param nspr: the NSPR to which the VNF belongs
        :return: dict of the VLs connected to the specified VNF
        """
        vnf_links = {}
        for extremes, vl in nspr.edges.items():
            if vnf_id in extremes:
                vnf_links[extremes] = vl
        return vnf_links

    def compute_link_weight(self, source: int, target: int, link: dict):
        """ Compute the weight of an edge between two nodes.
        If the edge satisfies the bandwidth requirement, the weight is 1, else infinity.

        This method is passed to networkx's shortest_path function as a weight function, and it's subject to networkx's API.
        It must take exactly 3 arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge.
        We need the required bandwidth to compute an edge's weight, so we save it into an attribute of the simulator (self._cur_vl_reqBW).

        :param source: source node in the PSN
        :param target: target node in the PSN
        :param link: dict of the link's (source - target) attributes
        :return: the weight of that link
        """
        return 1 if link['availBW'] >= self._cur_vl_reqBW else math.inf

    def _init_obs_dict(self) -> dict:
        """
        Initialize the observation dict.

        To be called after reading a PSN and before placing any VNF/VL on it.
        """
        # check that the env has a PSN
        try:
            if self.psn is None:
                raise ValueError("self.psn is None")
        except AttributeError:
            raise AttributeError("self.psn is not defined")

        # initialize lists
        cpu_avails = np.zeros(len(self.psn.nodes), dtype=np.float32)
        ram_avails = np.zeros(len(self.psn.nodes), dtype=np.float32)
        bw_avails = np.zeros(len(self.psn.nodes), dtype=np.float32)
        placement_state = np.zeros(len(self.psn.nodes), dtype=int)

        # scan all nodes and save data in lists
        self.tot_cpu_cap = self.tot_ram_cap = self.tot_bw_cap = 0
        for node_id, node in self.psn.nodes.items():
            self.tot_cpu_cap += node.get('CPUcap', 0)
            self.tot_ram_cap += node.get('RAMcap', 0)
            cpu_avails[self.map_id_idx[node_id]] = node.get('availCPU', 0)
            ram_avails[self.map_id_idx[node_id]] = node.get('availRAM', 0)
        # scan all links and save data in list
        for extremes, link in self.psn.edges.items():
            self.tot_bw_cap += link['BWcap']
            bw_avails[self.map_id_idx[extremes[0]]] += link['availBW']
            bw_avails[self.map_id_idx[extremes[1]]] += link['availBW']

        # save max CPU/RAM/BW capacities (= availabilities in empty PSN) of all nodes
        self.max_cpu = np.max(cpu_avails)
        self.max_ram = np.max(ram_avails)
        self.max_bw = np.max(bw_avails)

        # normalize the quantities
        cpu_avails /= self.max_cpu
        ram_avails /= self.max_ram
        bw_avails /= self.max_bw

        obs = {
            # PSN state
            'cpu_avails': cpu_avails,
            'ram_avails': ram_avails,
            'bw_avails': bw_avails,
            'placement_state': placement_state,
            # NSPR state
            'cur_vnf_cpu_req': np.array([0], dtype=int),
            'cur_vnf_ram_req': np.array([0], dtype=int),
            'cur_vnf_bw_req': np.array([0], dtype=int),
            'vnfs_still_to_place': np.array([0], dtype=int)
        }

        # store the obs for an empty PSN
        del self._empty_psn_obs_dict
        self._empty_psn_obs_dict = copy.deepcopy(obs)

        return obs

    def update_nspr_state(self) -> GymObs:
        """ Get an observation from the environment.

        The PSN state is already dynamically kept updated, so this method
        will only collect data about the NSPR state and complete the observation
        dict, that will be returned.

        :return: an instance of an observation from the environment
        """
        # state regarding the NSPR
        if self.cur_vnf is not None:
            cur_vnf_vls = self.get_cur_vnf_vls(vnf_id=self.cur_vnf_id,
                                               nspr=self.cur_nspr)
            cur_vnf_cpu_req = np.array(
                [self.cur_vnf['reqCPU'] / self.max_cpu], dtype=np.float32)

            cur_vnf_ram_req = np.array(
                [self.cur_vnf['reqRAM'] / self.max_ram], dtype=np.float32)

            cur_vnf_bw_req = np.array(
                [sum([vl['reqBW'] for vl in cur_vnf_vls.values()]) / self.max_bw],
                dtype=np.float32)

            vnfs_still_to_place = np.array(
                [len(self.cur_nspr_unplaced_vnfs_ids) + 1], dtype=int)
        else:
            cur_vnf_cpu_req = np.array([0], dtype=np.float32)
            cur_vnf_ram_req = np.array([0], dtype=np.float32)
            cur_vnf_bw_req = np.array([0], dtype=np.float32)
            vnfs_still_to_place = np.array([0], dtype=int)

        self.obs_dict['cur_vnf_cpu_req'] = cur_vnf_cpu_req
        self.obs_dict['cur_vnf_ram_req'] = cur_vnf_ram_req
        self.obs_dict['cur_vnf_bw_req'] = cur_vnf_bw_req
        self.obs_dict['vnfs_still_to_place'] = vnfs_still_to_place
        return self.obs_dict

    def reset(self, **kwargs) -> GymObs:
        """ Method used to reset the environment

        :return: the starting/initial observation of the environment
        """
        self.done = False   # re-set 'done' attribute

        # if last NSPR has not been placed completely, remove it, this is a new episode
        self.cur_nspr = None

        # reset network status (simply re-read the PSN file)
        # (needed because the available BW of the links gets actually modified)
        self.psn = reader.read_psn(graphml_file=self.psn_file)

        self.ep_number += 1
        self.nsprs_seen_in_cur_ep = 0

        # read the NSPRs to be evaluated
        # self.nsprs = reader.read_nsprs(nsprs_path=self.nsprs_path)
        self.nsprs = reader.sample_nsprs(nsprs_path=self.nsprs_path,
                                         n=self.nsprs_per_episode,
                                         min_arrival_time=self.time_step,
                                         max_duration=self.nsprs_max_duration)

        # reset partial rewards to be accumulated across the episodes' steps
        self.reset_partial_rewards()

        # return the obs corresponding to an empty PSN:
        # ALTERNATIVE 1: slower, but runs through the network and works with changing PSNs
        # self._obs_dict = self._init_obs_dict()

        # ALTERNATIVE 2: slightly faster on paper, but does not work with changing PSNs
        del self.obs_dict
        self.obs_dict = copy.deepcopy(self._empty_psn_obs_dict)

        # get arrived NSPRs
        self.waiting_nsprs += self.nsprs.get(self.time_step, [])
        self.pick_next_nspr()

        # update action mask (if no action masking is implemented, it has no effect)
        self._action_mask[:] = True
        # verison one: more randomic
        # indexes = np.random.rand(*self._action_mask.shape) < self.perc_avail_nodes
        # version two: less randomic
        size = round((1. - self.perc_avail_nodes) * self.action_space.n)
        indexes = np.random.choice(self.action_space.n, size=size, replace=False)
        self._action_mask[indexes] = False

        # new observation
        obs = self.update_nspr_state()

        return obs

    def step(self, action: int) -> Tuple[GymObs, float, bool, dict]:
        """ Perform an action in the environment

        :param action: the action to be performed
            more in detail, it's the index in the list of server corresponding
            ot a certain server ID, the mapping between this index and the
            server ID is done in the self.servers_map_idx_id dictionary
        :return: next observation, reward, done (True if the episode is over), info
        """
        reward, info = 0, {}

        # this happens only when the agent is prevented from choosing nodes that don't have enough resources,
        # i.e., when the environment is wrapped with PreventInfeasibleActions
        # if action < 0:
        #     obs, reward = self.manage_unsuccessful_action()
        #     return obs, reward, done, info

        # place the VNF and update the resources availabilities of the physical node
        if self.cur_nspr is not None:
            physical_node_id = self.servers_map_idx_id[action]
            physical_node = self.psn.nodes[physical_node_id]

            if not self.enough_avail_resources(physical_node_id, self.cur_vnf):
                # the VNF cannot be placed on the physical node
                obs, reward = self.manage_unsuccessful_action()
                return obs, reward, self.done, info

            # update acceptance reward and tr_load balancing reward
            idx = self.map_id_idx[physical_node_id]
            self._acceptance_rewards.append(self.rval_accepted_vnf)
            self._load_balance_rewards.append(
                self.obs_dict['cpu_avails'][idx] * self.max_cpu / physical_node['CPUcap'] +
                self.obs_dict['ram_avails'][idx] * self.max_ram / physical_node['RAMcap']
            )

            # update the resources availabilities of the physical node in the obs dict
            self.cur_vnf['placed'] = physical_node_id
            self.obs_dict['cpu_avails'][idx] -= self.cur_vnf['reqCPU'] / self.max_cpu
            self.obs_dict['ram_avails'][idx] -= self.cur_vnf['reqRAM'] / self.max_ram
            self.obs_dict['placement_state'][idx] += 1

            # connect the placed VNF to the other VNFs it's supposed to be connected to
            cur_vnf_VLs = self.get_cur_vnf_vls(self.cur_vnf_id, self.cur_nspr)
            if not cur_vnf_VLs:
                # if the VNF is detached from all others, R.C. reward is 1,
                # so it's the neutral when aggregating the rewards into the global one
                self._resource_consumption_rewards.append(1)
            else:
                for (source_vnf, target_vnf), vl in cur_vnf_VLs.items():
                    # get the physical nodes where the source and target VNFs are placed
                    source_node = self.cur_nspr.nodes[source_vnf]['placed']
                    target_node = self.cur_nspr.nodes[target_vnf]['placed']

                    # if the VL isn't placed yet and both the source and target VNFs are placed, place the VL
                    if not vl['placed'] and source_node >= 0 and target_node >= 0:
                        self._cur_vl_reqBW = vl['reqBW']
                        psn_path = nx.shortest_path(G=self.psn,
                                                    source=source_node,
                                                    target=target_node,
                                                    weight=self.compute_link_weight,
                                                    method='dijkstra')

                        """ if NO path is available, 'nx.shortest_path' will
                        return an invalid path. Only after the whole VL has been
                        placed, it is possible to restore the resources
                        availabilities, so we use this variable to save that the
                        resources have been exceeded as soon as we find this to
                        happen, and only after the VL placement, if this var is
                        True, we restore the resources availabilities. """
                        exceeded_bw = False
                        # place VL onto the PSN
                        # and update the resources availabilities of physical links involved
                        for i in range(len(psn_path) - 1):
                            physical_link = self.psn.edges[psn_path[i], psn_path[i + 1]]
                            extreme1_idx = self.map_id_idx[psn_path[i]]
                            extreme2_idx = self.map_id_idx[psn_path[i + 1]]
                            self.obs_dict['bw_avails'][extreme1_idx] -= vl['reqBW'] / self.max_bw
                            self.obs_dict['bw_avails'][extreme2_idx] -= vl['reqBW'] / self.max_bw
                            # note: here the PSN is actually modified: the available
                            # BW of the link is decreased. Needed for shortest path computation
                            physical_link['availBW'] -= vl['reqBW']
                            if physical_link['availBW'] < 0:
                                exceeded_bw = True
                        vl['placed'] = psn_path

                        if exceeded_bw:
                            obs, reward = self.manage_unsuccessful_action()
                            return obs, reward, self.done, info

                        # update the resource consumption reward
                        path_length = len(psn_path) - 1
                        self._cur_resource_consumption_rewards.append(
                            1 / path_length if path_length > 0 else 1)

                # aggregate the resource consumption rewards into a single value for this action
                n_VLs_placed_now = len(self._cur_resource_consumption_rewards)
                if n_VLs_placed_now == 0:
                    self._resource_consumption_rewards.append(1.)
                else:
                    self._resource_consumption_rewards.append(
                        sum(self._cur_resource_consumption_rewards) / n_VLs_placed_now)
                    self._cur_resource_consumption_rewards = []

            # save the ID of the next VNF
            if self.cur_nspr_unplaced_vnfs_ids:
                self.cur_vnf_id = self.cur_nspr_unplaced_vnfs_ids.pop(0)
                if self.accumulate_reward:
                    reward = 0  # global reward is non-zero only after the whole NSPR is placed (as HADRL)
                else:
                    # eventual discount factor of the acceptance reward
                    if self.discount_acc_rew:
                        self.acc_rew_disc_fact += self.base_acc_rew_disc_fact
                    else:
                        self.acc_rew_disc_fact = 1.
                    # reward always givent to the agent
                    reward = self._acceptance_rewards[-1] * self.acc_rew_disc_fact * \
                             self._load_balance_rewards[-1] * \
                             self._resource_consumption_rewards[-1] / len(self.cur_nspr.nodes) / \
                             10.    # scaling factor
                reward = self._normalize_reward_0_10(reward)
            else:
                # it means we finished the VNFs of the current NSPR
                self.nsprs_seen_in_cur_ep += 1
                self.tot_seen_nsprs += 1
                if self.nsprs_seen_in_cur_ep >= self.nsprs_per_episode:
                    self.done = True
                # reset placement state
                self.obs_dict['placement_state'] = np.zeros(len(self.psn.nodes), dtype=int)
                # update global reward because the NSPR is fully placed
                reward = np.stack((self._acceptance_rewards,
                                   self._resource_consumption_rewards,
                                   self._load_balance_rewards)).prod(axis=0).sum()
                # normalize the reward to be in [0, 10] (as they do in HA-DRL)
                reward = self._normalize_reward_0_10(reward) * \
                         2  # TODO: per dargli più peso (non da HADRL)
                self.reset_partial_rewards()
                self.cur_nspr = None    # marked as None so a new one can be picked
                # update the acceptance ratio
                self.accepted_nsprs += 1

        # increase time step
        self.time_step += 1

        # check for new and departing NSPRs
        if self.nsprs is not None:
            self.check_for_departed_nsprs()
            self.waiting_nsprs += self.nsprs.get(self.time_step, [])
            self.pick_next_nspr()

        # new observation
        obs = self.update_nspr_state()

        return obs, reward, self.done, info

    def render(self, mode="human"):
        raise NotImplementedError
