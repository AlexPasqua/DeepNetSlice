import math
from typing import Union, Tuple

import src.spaces
from spaces.enhanced_discrete import EnhancedDiscrete

import gym
import networkx as nx
import numpy as np

import reader

GymObs = Union[Tuple, dict, np.ndarray, int]


class Simulator(gym.Env):
    """ Class implementing the network simulator (RL environment)

    Attributes:
        psn (nx.Graph): physical substrate network
        nsprs (dict): dict of NSPRs associated to their arrival time
        decision_maker (DecisionMaker): decision maker used to decide the next VNF to place onto the PSN
    """

    def __init__(self, psn_file: str, nsprs_path: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        """
        super(Simulator, self).__init__()

        self.psn = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs_path = nsprs_path  # path to the directory containing the NSPRs
        self.nsprs = None  # will be initialized in the reset method

        # attributes needed in the method 'step' because it has no access to observations
        self.cur_nspr = None  # used to keep track of the current NSPR being evaluated
        self.cur_nspr_unplaced_vnfs_ids = []  # used to keep track of the VNFs' IDs of the current NSPR that haven't been placed yet
        self.cur_vnf_id = None  # used to keep track of the current VNF being evaluated

        self._cur_vl_reqBW = 0  # auxiliary attribute needed in method 'self.compute_link_weight'

        # map (dict) between IDs of PSN's nodes and their respective index (see self._init_map_id_idx's docstring)
        nodes_ids = list(self.psn.nodes.keys())
        self._map_id_idx = {nodes_ids[idx]: idx for idx in range(len(nodes_ids))}

        # map (dict) between an index of a list (incrementing int) and the ID of a server
        servers_ids = [node_id for node_id, node in self.psn.nodes.items() if node['NodeType'] == 'server']
        self._servers_map_idx_id = {idx: servers_ids[idx] for idx in range(len(servers_ids))}

        # partial rewards to be accumulated across the steps of evaluation of a single NSPR
        self._acceptance_rewards = []
        self._resource_consumption_rewards = []
        self._cur_resource_consumption_rewards = []
        self._load_balancing_rewards = []

        # reward values for specific outcomes
        self.rval_accepted_vnf = 100
        self.rval_rejected_vnf = -100

        # Action space and observation space (gym.Env required attributes)
        ONE_BILLION = 1000000000  # constant for readability
        n_nodes = len(self.psn.nodes)
        # action space = number of servers
        self.action_space = src.spaces.EnhancedDiscrete(len(servers_ids) + 1, start=-1)
        self.observation_space = gym.spaces.Dict({
            # PSN STATE
            'cpu_capacities': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
            'ram_capacities': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
            # for each physical node, sum of the BW of the physical links connected to it
            'bw_capacity_per_node': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
            # for each physical node, number of VNFs of the current NSPR placed on it
            'placement_state': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(n_nodes,), dtype=int),

            # NSPR STATE
            # note: apparently it's not possible to pass "math.inf" or "sys.maxsize" as a gym.spaces.Box's high value
            'cur_vnf_cpu_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
            'cur_vnf_ram_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
            # sum of the required BW of each VL connected to the current VNF
            'cur_vnf_bw_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
            'vnfs_still_to_place': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
        })

    def _get_observation(self, cur_vnf: Union[dict, None]) -> GymObs:
        """ Method used to get the observation of the environment.

        :param cur_vnf: current VNF being evaluated
        :return: an instance of an observation from the environment
        """
        # initialize lists
        cpu_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        ram_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        bw_capacity_per_node = np.zeros(len(self.psn.nodes), dtype=np.float32)
        placement_state = np.zeros(len(self.psn.nodes), dtype=int)

        # TODO: this could probably be made more efficient
        # scan all nodes and save data in lists
        for node_id, node in self.psn.nodes.items():
            # get nodes' capacities (if routers, set these to 0)
            cpu_capacities[self._map_id_idx[node_id]] = node.get('CPUcap', 0)
            ram_capacities[self._map_id_idx[node_id]] = node.get('RAMcap', 0)
            for extremes, link in self.psn.edges.items():
                if node_id in extremes:
                    bw_capacity_per_node[self._map_id_idx[node_id]] += link['availBW']

        # state regarding the NSPR
        if cur_vnf is not None:
            cur_vnf_vls = self.get_cur_vnf_vls(vnf_id=self.cur_vnf_id, nspr=self.cur_nspr)
            nspr_state = {'cur_vnf_cpu_req': np.array([cur_vnf['reqCPU']], dtype=int),
                          'cur_vnf_ram_req': np.array([cur_vnf['reqRAM']], dtype=int),
                          'cur_vnf_bw_req': np.array([sum(vl['reqBW'] for vl in cur_vnf_vls.values())], dtype=int),
                          'vnfs_still_to_place': np.array([len(self.cur_nspr_unplaced_vnfs_ids)], dtype=int)}
        else:
            nspr_state = {'cur_vnf_cpu_req': np.array([0], dtype=int), 'cur_vnf_ram_req': np.array([0], dtype=int),
                          'cur_vnf_bw_req': np.array([0], dtype=int), 'vnfs_still_to_place': np.array([0], dtype=int)}

        # instance of an observation from the environment
        obs = {
            'cpu_capacities': cpu_capacities,
            'ram_capacities': ram_capacities,
            'bw_capacity_per_node': bw_capacity_per_node,
            'placement_state': placement_state,
            **nspr_state
        }
        return obs

    def reset_partial_rewards(self):
        """ Resets the partial rewards (used in case a NSPR cannot be placed) """
        self._acceptance_rewards = []
        self._resource_consumption_rewards = []
        self._load_balancing_rewards = []

    def reset(self) -> GymObs:
        """ Method used to reset the environment

        :return: the starting/initial observation of the environment
        """
        # read the NSPRs to be evaluated
        self.nsprs = reader.read_nsprs(nsprs_path=self.nsprs_path)

        # reset partial rewards to be accumulated across the episodes' steps
        self._acceptance_rewards = []
        self._resource_consumption_rewards = []
        self._load_balancing_rewards = []

        # Save the first NSPR as current one to be evaluated
        # TODO: self.nsprs is a dict with the arrival time as key and the list of VNFs as value.
        #       Currently, we don't consider the arrival time during the agent's training,
        #       so we just take the 'values' of the dict.
        #       In the future we need to decide whether to consider the arrival time or not during training (and inference) and organize better the NSPRs.
        for arrival_time, nspr in self.nsprs.items():
            if self.nsprs[arrival_time]:
                self.cur_nspr = self.nsprs[arrival_time].pop(0)
                break

        cur_vnf = None
        if self.cur_nspr is not None:
            self.cur_nspr_unplaced_vnfs_ids = list(self.cur_nspr.nodes.keys())
            self.cur_vnf_id = self.cur_nspr_unplaced_vnfs_ids.pop(0)
            cur_vnf = self.cur_nspr.nodes[self.cur_vnf_id]

        obs = self._get_observation(cur_vnf=cur_vnf)
        return obs

    def step(self, action) -> Tuple[GymObs, float, bool, dict]:
        """ Perform an action in the environment

        :param action: the action to be performed
            more in detail, it's the index in the list of server corresponding ot a certain server ID,
            the mapping between this index and the server ID is done in the self._servers_map_idx_id dictionary
        :return: next observation, reward, done (True if the episode is over), info
        """
        reward = 0
        done = False
        info = {}

        if action < 0:
            # it wasn't possible to place the VNF
            reward = self.rval_rejected_vnf
            done = True
            self.restore_avail_resources(nspr=self.cur_nspr)
        else:
            physical_node_id = self._servers_map_idx_id[action]

            # place the VNF and update the resources availabilities of the physical node
            if self.cur_nspr is not None:
                cur_vnf = self.cur_nspr.nodes[self.cur_vnf_id]
                physical_node = self.psn.nodes[physical_node_id]

                # update the resources availabilities of the physical node
                cur_vnf['placed'] = physical_node_id
                physical_node['availCPU'] -= cur_vnf['reqCPU']
                physical_node['availRAM'] -= cur_vnf['reqRAM']

                # update acceptance reward and load balancing reward
                self._acceptance_rewards.append(self.rval_accepted_vnf)
                self._load_balancing_rewards.append(
                    physical_node['availCPU'] / physical_node['CPUcap'] +
                    physical_node['availRAM'] / physical_node['RAMcap']
                )

                # connect the placed VNF to the other VNFs it's supposed to be connected to
                cur_vnf_VLs = self.get_cur_vnf_vls(self.cur_vnf_id, self.cur_nspr)  # get the VLs involving the current VNF
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
                            psn_path = nx.shortest_path(G=self.psn, source=source_node, target=target_node,
                                                        weight=self.compute_link_weight, method='dijkstra')

                            # place the VL onto the PSN and update the resources availabilities of the physical links involved
                            for i in range(len(psn_path) - 1):
                                physical_link = self.psn.edges[psn_path[i], psn_path[i + 1]]
                                physical_link['availBW'] -= vl['reqBW']
                            vl['placed'] = psn_path

                            # update the resource consumption reward
                            path_length = len(psn_path) - 1
                            self._cur_resource_consumption_rewards.append(1 / path_length if path_length > 0 else 1)

                    # aggregate the resource consumption rewards into a single value for this action
                    n_VLs_placed_now = len(self._cur_resource_consumption_rewards)
                    if n_VLs_placed_now == 0:
                        self._resource_consumption_rewards.append(1)
                    else:
                        self._resource_consumption_rewards.append(
                            sum(self._cur_resource_consumption_rewards) / n_VLs_placed_now
                        )
                        self._cur_resource_consumption_rewards = []

                # save the ID of the next VNF
                if self.cur_nspr_unplaced_vnfs_ids:
                    self.cur_vnf_id = self.cur_nspr_unplaced_vnfs_ids.pop(0)
                    reward = 0  # global reward is non-zero only after the whole NSPR is placed
                else:
                    # it means we finished the VNFs of the current NSPR, so...
                    # update global reward because the NSPR is fully placed
                    reward = np.stack((self._acceptance_rewards,
                                       self._resource_consumption_rewards,
                                       self._load_balancing_rewards)).prod(axis=0).sum()
                    # reset partial rewards
                    self.reset_partial_rewards()

                    # pick next NSPR
                    self.cur_nspr = None
                    for arrival_time, nspr in self.nsprs.items():
                        if self.nsprs[arrival_time]:
                            self.cur_nspr = self.nsprs[arrival_time].pop(0)
                            break
                    if self.cur_nspr is not None:
                        self.cur_nspr_unplaced_vnfs_ids = list(self.cur_nspr.nodes.keys())
                        self.cur_vnf_id = self.cur_nspr_unplaced_vnfs_ids.pop(0)
                    else:
                        # it means we finished all the NSPRs
                        done = True

        # new observation
        if self.cur_nspr is not None:
            obs = self._get_observation(cur_vnf=self.cur_nspr.nodes[self.cur_vnf_id])
        else:
            obs = self._get_observation(cur_vnf=None)

        return obs, reward, done, info

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

    def restore_avail_resources(self, nspr: nx.Graph):
        """ Method called in case a NSPR is not accepted.
        Restores the resources if the PSN that had been already allocated for the rejected NSPR

        :param nspr: the rejected NSPR
        """
        for vnf_id, vnf in nspr.nodes.items():
            # restore nodes' resources availabilities
            if vnf['placed'] >= 0:
                physical_node = self.psn.nodes[vnf['placed']]
                physical_node['availCPU'] += vnf['reqCPU']
                physical_node['availRAM'] += vnf['reqRAM']
        for _, vl in nspr.edges.items():
            # restore links' resources availabilities
            if vl['placed']:
                # if vl['placed'] is not empty, it's the list of the physical nodes traversed by the link
                for i in range(len(vl['placed']) - 1):
                    physical_link = self.psn.edges[vl['placed'][i], vl['placed'][i + 1]]
                    physical_link['availBW'] += vl['reqBW']
        # reset partial rewards
        self.reset_partial_rewards()
