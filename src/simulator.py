import math
import sys

import gym

import networkx as nx
import numpy as np

from decision_makers import decision_makers
import reader


class Simulator(gym.Env):
    """ Class implementing the network simulator (RL environment)

    Attributes:
        psn (nx.Graph): physical substrate network
        nsprs (dict): dict of NSPRs associated to their arrival time
        decision_maker (DecisionMaker): decision maker used to decide the next VNF to place onto the PSN
    """

    def __init__(self, psn_file: str, nsprs_path: str, decision_maker_type: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        :param decision_maker_type: type of decision maker
        """
        self.psn = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs = reader.read_nsprs(nsprs_path=nsprs_path)  # network slice placement requests
        self.decision_maker = decision_makers[decision_maker_type]
        self.cur_nspr = None    # used to keep track of the current NSPR being evaluated
        self._cur_vl_reqBW = 0  # attribute needed in method 'self.compute_link_weight'

        # map (dict) between IDs of PSN's nodes and their respective index (see self._init_map_id_idx's docstring)
        self._map_id_idx = self._init_map_id_idx()

        # partial rewards to be accumulated across the episodes' steps
        self._acceptance_reward = 0
        self._resource_consumption_reward = 0
        self._load_balancing_reward = 0

        # gym.Env required attributes
        ONE_BILLION = 1000000000    # constant for readability
        n_nodes = len(self.psn.nodes)
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Dict({
            'psn_state': gym.spaces.Dict({
                'cpu_capacities': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
                'ram_capacities': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
                # for each physical node, sum of the BW of the physical links connected to it
                'bw_capacity_per_node': gym.spaces.Box(low=0, high=math.inf, shape=(n_nodes,), dtype=np.float32),
                # for each physical node, number of VNFs of the current NSPR placed on it
                'placement_state': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(n_nodes,), dtype=int),
            }),
            'nspr_state': gym.spaces.Dict({
                # note: apparently it's not possible to pass "math.inf" or "sys.maxsize" as a gym.spaces.Box's high value
                'cur_vnf_cpu_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
                'cur_vnf_ram_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
                # sum of the required BW of each VL connected to the current VNF
                'cur_vnf_bw_req': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
                'vnfs_still_to_place': gym.spaces.Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
            })
        })

    # def _get_max_vnfs_in_nspr(self):
    #     """ Auxiliary method to get the maximum number of VNFs that a single NSPR contains.
    #     Used to initialized the placement state in the observation space of the environment.
    #     """
    #     max_vnfs_in_nspr = 0
    #     for arrival_time, cur_vnfs in self.nsprs.items():
    #         for vnf in cur_vnfs:
    #             if len(vnf.nodes) > max_vnfs_in_nspr:
    #                 max_vnfs_in_nspr = len(vnf.nodes)
    #     return max_vnfs_in_nspr

    def _init_map_id_idx(self):
        """ Method used to initialize a map between the IDs of the physical node and an integer,
        such that a bunch of N IDs gets mapped to a bunch of N successive integers from 0 to N-1.
        e.g. IDs: 0, 1, 3, 9 -> IDs_map: 0:0, 1:1, 3:2, 9:3.
        This is used because the simulator needs to have some lists where each element corresponds to a node,
        so we need a way to know which index corresponds to which node.

        :return: dict mapping the IDs to the corresponding integer
        """
        ids_map, idx = {}, 0
        for node_id, node in self.psn.nodes.items():
            ids_map[node_id] = idx
            idx += 1
        return ids_map

    # TODO: probably remove this method (probabily useless and there are some mistakes)
    def _get_observation(self, reset=False):
        """ Method used to get the observation of the environment.

        :param reset: if True, the observation is the reset state of the environment

        :return: an instance of an observation from the environment
        """
        # initialize lists
        cpu_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        ram_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        bw_capacity_per_node = np.zeros(len(self.psn.nodes), dtype=np.float32)
        placement_state = np.zeros(len(self.psn.nodes), dtype=int)

        # scan all nodes and save data in lists
        for node_id, node in self.psn.nodes.items():
            cpu_capacities[self._map_id_idx[node_id]] = node['CPUcap']
            ram_capacities[self._map_id_idx[node_id]] = node['RAMcap']
            for extremes, link in self.psn.edges.items():
                if node_id in extremes:
                    bw_capacity_per_node[self._map_id_idx[node_id]] += link['availBW']

        # if this method is called form the reset method, set this part of the observation with zeros
        nspr_state = {'cur_vnf_cpu_req': np.array([0]), 'cur_vnf_ram_req': np.array([0]),
                      'cur_vnf_bw_req': np.array([0]), 'vnfs_still_to_place': np.array([0])}

        return {
            'psn_state': {
                'cpu_capacities': cpu_capacities,
                'ram_capacities': ram_capacities,
                'bw_capacity_per_node': bw_capacity_per_node,
                'placement_state': placement_state,
            },
            'nspr_state': nspr_state
        }

    def reset(self):
        """ Method used to reset the environment

        :return: the starting/initial observation of the environment
        """
        # reset partial rewards to be accumulated across the episodes' steps
        self._acceptance_reward = 0
        self._resource_consumption_reward = 0
        self._load_balancing_reward = 0

        # initialize lists
        cpu_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        ram_capacities = np.zeros(len(self.psn.nodes), dtype=np.float32)
        bw_capacity_per_node = np.zeros(len(self.psn.nodes), dtype=np.float32)
        placement_state = np.zeros(len(self.psn.nodes), dtype=int)

        # scan all nodes and save data in lists
        for node_id, node in self.psn.nodes.items():
            cpu_capacities[self._map_id_idx[node_id]] = node['CPUcap']
            ram_capacities[self._map_id_idx[node_id]] = node['RAMcap']
            for extremes, link in self.psn.edges.items():
                if node_id in extremes:
                    bw_capacity_per_node[self._map_id_idx[node_id]] += link['availBW']

        # if there are NSPRs arriving at time t=0, save the first as current NSPR to be evaluated
        starting_nsprs = self.nsprs.get(0, [])
        if starting_nsprs:
            self.cur_nspr = starting_nsprs[0]
            cur_nspr_vnfs = list(self.cur_nspr.nodes.keys())
            cur_vnf_id = cur_nspr_vnfs[0]
            cur_vnf = self.cur_nspr.nodes[cur_vnf_id]
            cur_vnf_vls = self.get_cur_vnf_vls(vnf_id=cur_vnf_id, nspr=self.cur_nspr)
            nspr_state = {'cur_vnf_cpu_req': np.array([cur_vnf['reqCPU']], dtype=int),
                          'cur_vnf_ram_req': np.array([cur_vnf['reqRAM']], dtype=int),
                          'cur_vnf_bw_req': np.array([sum(vl['reqBW'] for vl in cur_vnf_vls.values())], dtype=int),
                          'vnfs_still_to_place': np.array([len(cur_nspr_vnfs)], dtype=int)}
        else:
            # there's no NSPR to be evaluated, so set the NSPR state to zeros
            nspr_state = {'cur_vnf_cpu_req': np.array([0]), 'cur_vnf_ram_req': np.array([0]),
                          'cur_vnf_bw_req': np.array([0]), 'vnfs_still_to_place': np.array([0])}

        return {
            'psn_state': {
                'cpu_capacities': cpu_capacities,
                'ram_capacities': ram_capacities,
                'bw_capacity_per_node': bw_capacity_per_node,
                'placement_state': placement_state,
            },
            'nspr_state': nspr_state
        }

    def step(self, action):
        physical_node_id = action
        reward = 0
        done = False
        info = {}

        if physical_node_id < 0:
            # it wasn't possible to place the VNF
            self._acceptance_reward = -100
            done = True
            self.restore_avail_resources(nspr=self.cur_nspr)
        else:
            # place the VNF and update the resources availabilities of the physical node
            # self.psn.nodes[physical_node_id]['availCPU'] -= self.
            pass



        if done:
            reward = self._acceptance_reward + self._resource_consumption_reward + self._load_balancing_reward

        return self._get_observation(reset=False), reward, done, info

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

    def evaluate_nspr(self, nspr: nx.Graph) -> bool:
        """ Place all the VNFs and VLs onto the physical network and update the available resources

        :param nspr: a NSPR to be evaluated (accepted and placed / reject)
        :return: True if the NSPR is accepted and placed on the PSN, else False
        """
        for vnf_id, vnf in nspr.nodes.items():
            if vnf['placed'] < 0:  # it means the VNF is not currently placed onto a physical node
                # select the physical node onto which to place the VNF
                physical_node_id, physical_node = self.decision_maker.decide_next_node(psn=self.psn, vnf=vnf)
                self.step(action=physical_node_id)

                # place the VNF and update the resources availabilities of the physical node
                vnf['placed'] = physical_node_id
                physical_node['availCPU'] -= vnf['reqCPU']
                physical_node['availRAM'] -= vnf['reqRAM']

                # connect the placed VNF to the other VNFs it's supposed to be connected to
                cur_vnf_VLs = self.get_cur_vnf_vls(vnf_id, nspr)  # get the VLs involving the current VNF
                for (source_vnf, target_vnf), vl in cur_vnf_VLs.items():
                    # get the physical nodes where the source and target VNFs are placed
                    source_node, target_node = nspr.nodes[source_vnf]['placed'], nspr.nodes[target_vnf]['placed']

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

    def start(self, sim_steps: int = 100):
        """ Main cycle of the simulator

        :param sim_steps: number of simulation steps to be performed
        """
        current_nsprs = []
        for step in range(sim_steps):
            # add eventual newly arrived NSPRs to the list of NSPRs to be evaluated and skip if there are none
            current_nsprs += self.nsprs.get(step, [])
            if len(current_nsprs) == 0:
                continue

            # pop a NSPR from the list of NSPRs that arrived already
            cur_nspr = current_nsprs.pop(0)

            # accept/reject NSPR
            outcome = self.evaluate_nspr(nspr=cur_nspr)
            if not outcome:
                self.restore_avail_resources(nspr=cur_nspr)
