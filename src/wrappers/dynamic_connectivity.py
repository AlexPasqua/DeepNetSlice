import gym


class DynamicConnectivity(gym.Wrapper):
    """ Changes the connectivity of the PSN episode by episode """
    
    def __init__(self, env):
        super().__init__(env)
        self.psn = None
        self.tot_bw_cap = None
    
    def reset(self, **kwargs):
        self.psn = self.unwrapped.psn
        if self.tot_bw_cap is None:
            self.tot_bw_cap = self.get_tot_bw_cap()
        self.psn = self.remove_all_edges()
        self.psn = self.add_edges()
    
    def get_tot_bw_cap(self):
        tot = 0
        for edge in self.psn.edges.values():
            tot += edge['BWcap']
        
        tot = [edge['BWcap'] for edge in self.psn.edges.values()].sum()
        pass

    def remove_all_edges(self):
        for (u, v), edge in self.psn.edges.items():
            self.psn.remove_edge(u, v)

    def add_edges(self):
        raise NotImplementedError
