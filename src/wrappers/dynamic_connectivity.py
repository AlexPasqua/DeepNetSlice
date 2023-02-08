import random
from typing import Callable, Optional
import gym
import networkx as nx
import numpy as np


class DynamicConnectivity(gym.Wrapper):
    """ Changes the connectivity of the PSN episode by episode """
    
    def __init__(
        self,
        env: gym.Env,
        link_bw: int = 10_000,
        nodes_mask: Optional[Callable[[gym.Env], np.ndarray]] = None
    ):
        """
        :param env: gym environment
        :param link_bw: total bandwidth capacity of each link
        :param nodes_mask: in not None, contains nodes to be removed form the PSN graph
        """
        super().__init__(env)
        self.nodes_mask = nodes_mask
        self.link_bw = link_bw
        self.tot_bw_cap = sum([edge['BWcap'] for edge in self.env.psn.edges.values()])
        self.placed_bw = 0
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # remove all edges from the PSN
        self.remove_all_edges()
        # eventually remove masked nodes
        if self.nodes_mask is not None:
            self.remove_masked_nodes()
        # initialize the bandwidth placed in the PSN
        self.placed_bw = 0
        # add edges in the PSN until the target bandwidth capacity is reached
        self.add_edges()
        return self.env.obs_dict    # updated in self.add_edges()
    
    def remove_all_edges(self):
        for u, v in self.env.psn.edges.keys():
            self.env.psn.remove_edge(u, v)
    
    def remove_masked_nodes(self):
        nodes_mask = self.nodes_mask(self.env)
        # indexes where the mask is False
        indexes_to_remove = np.where(np.logical_not(nodes_mask))[0]
        for idx in indexes_to_remove:
            node_id = self.env.servers_map_idx_id[idx]
            self.env.psn.remove_node(node_id)

    def add_edges(self):
        """Add edges to the PSN

        Chooses every time a random node an an unvisited node and connectes them.
        When no nodes are isolated, if the target BW hasn't been reached, it does so
        by adding further random links in the PSN.
        """
        # zero the BW availabilities in the obs dict
        self.env.obs_dict['bw_avails'] = np.zeros_like(self.env.obs_dict['bw_avails'])
        # set of unvisited nodes
        unvisited = set(self.env.psn.nodes)
        while unvisited:
            # sample a node form the PSN
            u = random.choice(list(self.env.psn.nodes))
            # sample an unvisited nodes to connect to it
            v = random.choice(list(unvisited))
            if u != v:
                # connect the 2 nodes
                self.env.psn.add_edge(u, v, BWcap=self.link_bw, availBW=self.link_bw)
                # save the amount of bandwidth introduced in the PSN
                self.placed_bw += self.link_bw
                # get the 2 nodes' indexes in the obs dict and update the obs dict
                u_idx = self.env.map_id_idx[u]
                v_idx = self.env.map_id_idx[v]
                self.env.obs_dict['bw_avails'][u_idx] += self.link_bw
                self.env.obs_dict['bw_avails'][v_idx] += self.link_bw
                # remove the nodes from the set of unvisited nodes
                unvisited.remove(v)
                if u in unvisited:
                    unvisited.remove(u)

        # if the total bandwidth of the PSN hasn't been reached, reach it by adding random links
        while self.placed_bw < self.tot_bw_cap:
            u, v = random.sample(self.env.psn.nodes, 2)
            # check that the 2 nodes aren't connected already
            if (u, v) not in self.env.psn.edges:
                bw = min(self.link_bw, self.tot_bw_cap - self.placed_bw)
                self.env.psn.add_edge(u, v, BWcap=bw, availBW=bw)
                self.placed_bw += bw
                # get the 2 nodes' indexes in the obs dict and update the obs dict
                u_idx = self.env.map_id_idx[u]
                v_idx = self.env.map_id_idx[v]
                self.env.obs_dict['bw_avails'][u_idx] += self.link_bw
                self.env.obs_dict['bw_avails'][v_idx] += self.link_bw
        
        # normalize the BW availabilities in the obs dict
        self.env.obs_dict['bw_avails'] /= np.max(self.env.obs_dict['bw_avails'])