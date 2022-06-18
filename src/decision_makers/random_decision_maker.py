import networkx as nx


class RandomDecisionMaker:
    """
    Random decision maker: when deciding a physical node where to place a VNF,
    it chooses one at random that satisfies the resources requirements
    """
    def __init__(self):
        pass

    def decide_next_node(self, psn: nx.Graph, nspr: nx.Graph):
        """
        Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet.
        For RandomDecisionMaker, it simply chooses at random a physical node that satisfies the resources requirements

        :param psn: a graph representation of the PSN
        :param nspr: a graph representation of a NSPR
        :return:
        """
        pass
