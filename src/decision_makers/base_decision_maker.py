import networkx as nx
from abc import abstractmethod


class BaseDecisionMaker:
    """ Base decision maker. All other decision makers need to inherit from this class """
    def __init__(self):
        pass

    @abstractmethod
    def decide_next_node(self, psn: nx.Graph, nspr: nx.Graph):
        """ Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet

        :param psn: a graph representation of the PSN
        :param nspr: a graph representation of a NSPR
        :raise NotImplementedError: because this is an abstract method that need to be overridden in sub-classes
        """
        raise NotImplementedError
