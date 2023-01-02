import random
import gym


class DynamicConnectivity(gym.Wrapper):
    """ Changes the connectivity of the PSN episode by episode """
    
    def __init__(self, env, link_bw: int = 10_000):
        super().__init__(env)
        self.link_bw = link_bw
        self.psn = self.unwrapped.psn
        self.tot_bw_cap = sum([edge['BWcap'] for edge in self.psn.edges.values()])
    
    def reset(self, **kwargs):
        self.psn = self.remove_all_edges()
        self.psn = self.add_edges()
    
    def remove_all_edges(self):
        for (u, v), edge in self.psn.edges.items():
            self.psn.remove_edge(u, v)

    def add_edges(self):
        raise NotImplementedError
        # # start connecting with DFS, so no node is isolated
        # unvisited = set(self.psn.nodes)
        # start = random.choice(unvisited)
        # visited = set(start)
        # unvisited.remove(start)
        # stack = [start]
        # while stack:
        #     current = stack[-1]
        #     unvisited_neighbors = [n for n in self.psn.neighbors(current) if n in unvisited]
        #     if not unvisited_neighbors:
        #         stack.pop()
        #     else:
        #         neighbor = random.choice(unvisited_neighbors)
        #         self.psn.add_edge(current, neighbor)
