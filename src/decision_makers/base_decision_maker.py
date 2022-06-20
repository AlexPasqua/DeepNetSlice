import networkx as nx
from abc import abstractmethod


class BaseDecisionMaker:
    """ Base decision maker. All other decision makers need to inherit from this class """
    def __init__(self):
        pass

    @staticmethod
    def resources_reqs_satisfied(physical_node: dict, vnf: dict):
        """ Checks whether the resources requirements of a certain VNF are satisfied by a certain physical node on the PSN

        :param physical_node: physical node on the PSN
        :param vnf: virtual network function requiring some resources
        :return: True if the resources required by the VNF are available on the physical node, else False
        """
        if physical_node['CPUcap'] >= vnf['reqCPU'] and physical_node['RAMcap'] >= vnf['reqRAM']:
            return True
        return False

    @abstractmethod
    def decide_next_node(self, psn: nx.Graph, nspr: nx.Graph):
        """ Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet

        :param psn: a graph representation of the PSN
        :param nspr: a graph representation of a NSPR
        :raise NotImplementedError: because this is an abstract method that need to be overridden in sub-classes
        """
        raise NotImplementedError
