import random
import gym


class DynamicConnectivity(gym.Wrapper):
    """ Changes the connectivity of the PSN episode by episode """
    
    def __init__(self, env, link_bw: int = 10_000):
        super().__init__(env)
        self.link_bw = link_bw
        self.psn = self.unwrapped.psn
        self.tot_bw_cap = sum([edge['BWcap'] for edge in self.psn.edges.values()])
        self.placed_bw = 0
    
    def reset(self, **kwargs):
        # remove all edges from the PSN
        self.remove_all_edges()
        # initialize the bandwidth placed in the PSN
        self.placed_bw = 0
        # add edges in the PSN until the target bandwidth capacity is reached
        self.add_edges()
        # TODO: check if needed -> se usa puntatori in teoria non serve
        self.unwrapped.psn = self.psn
    
    def remove_all_edges(self):
        for (u, v), edge in self.psn.edges.items():
            self.psn.remove_edge(u, v)

    def add_edges(self):
        """Add edges to the PSN

        Start connecting via DFS, to ensure that no node is isolated from the rest.
        Then, if the target bandwidth is not reached, reach it adding random links.
        """
        # set of unvisited nodes
        unvisited = set(self.psn.nodes)
        # starting node
        start = random.choice(list(self.psn.nodes))
        # set of visited nodes
        visited = {start}
        # remove 'start' from unvisited nodes
        unvisited.remove(start)
        # stack to hold the nodes in the current path
        stack = [start]
        while stack:
            # current node
            current = stack[-1]
            # If there are no unvisited nodes, backtrack
            if not unvisited:
                stack.pop()
            else:
                # choose a random unvisited node
                neighbor = random.choice(list(unvisited))
                if neighbor != current:
                    # connect the current node with the chosen neighbor
                    self.psn.add_edge(current, neighbor, BWcap=self.link_bw, availBW=self.link_bw)
                    # save the amount of bandwidth introduced in the PSN
                    self.placed_bw += self.link_bw
                    # add neighbor to the set of visited nodes
                    visited.add(neighbor)
                    # remove neighbor from the set of unvisited nodes
                    unvisited.remove(neighbor)
                    # push the neighbor onto the stack (next current node)
                    stack.append(neighbor)

        # if the total bandwidth of the PSN hasn't been reached, reach it by adding random links
        while self.placed_bw < self.tot_bw_cap:
            u, v = random.sample(self.psn.nodes, 2)
            bw = min(self.link_bw, self.tot_bw_cap - self.placed_bw)
            self.psn.add_edge(u, v, BWcap=bw, availBW=bw)
            self.placed_bw += bw