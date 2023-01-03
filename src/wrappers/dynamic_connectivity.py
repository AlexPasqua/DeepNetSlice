import random
import gym
import networkx as nx
import numpy as np


class DynamicConnectivity(gym.Wrapper):
    """ Changes the connectivity of the PSN episode by episode """
    
    def __init__(self, env, link_bw: int = 10_000):
        super().__init__(env)
        self.link_bw = link_bw
        self.psn = self.env.psn
        self.tot_bw_cap = sum([edge['BWcap'] for edge in self.psn.edges.values()])
        self.placed_bw = 0
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # remove all edges from the PSN
        self.remove_all_edges()
        # initialize the bandwidth placed in the PSN
        self.placed_bw = 0
        # add edges in the PSN until the target bandwidth capacity is reached
        self.add_edges()
        return self.env.obs_dict    # updated in self.add_edges()
    
    def remove_all_edges(self):
        for (u, v), edge in self.psn.edges.items():
            self.psn.remove_edge(u, v)

    def add_edges(self):
        """Add edges to the PSN

        Chooses every time a random node an an unvisited node and connectes them.
        When no nodes are isolated, if the target BW hasn't been reached, it does so
        by adding further random links in the PSN.
        """
        # zero the BW availabilities in the obs dict
        self.env.obs_dict['bw_avails'] = np.zeros_like(self.env.obs_dict['bw_avails'])
        # set of unvisited nodes
        unvisited = set(self.psn.nodes)
        while unvisited:
            # sample a node form the PSN
            u = random.choice(list(self.psn.nodes))
            # sample an unvisited nodes to connect to it
            v = random.choice(list(unvisited))
            if u != v:
                # connect the 2 nodes
                self.psn.add_edge(u, v, BWcap=self.link_bw, availBW=self.link_bw)
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
            u, v = random.sample(self.psn.nodes, 2)
            # check that the 2 nodes aren't connected already
            if (u, v) not in self.psn.edges:
                bw = min(self.link_bw, self.tot_bw_cap - self.placed_bw)
                self.psn.add_edge(u, v, BWcap=bw, availBW=bw)
                self.placed_bw += bw
                # get the 2 nodes' indexes in the obs dict and update the obs dict
                u_idx = self.env.map_id_idx[u]
                v_idx = self.env.map_id_idx[v]
                self.env.obs_dict['bw_avails'][u_idx] += self.link_bw
                self.env.obs_dict['bw_avails'][v_idx] += self.link_bw
        
        # normalize the BW availabilities in the obs dict
        self.env.obs_dict['bw_avails'] /= np.max(self.env.obs_dict['bw_avails'])