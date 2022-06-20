import copy
import random
import networkx as nx
from typing import Union

from .base_decision_maker import BaseDecisionMaker


class RandomDecisionMaker(BaseDecisionMaker):
    """
    Random decision maker: when deciding a physical node where to place a VNF,
    it chooses one at random that satisfies the resources requirements
    """
    def __init__(self):
        super().__init__()

    def decide_next_node(self, psn: nx.Graph, vnf: dict) -> Union[dict, None]:
        """
        Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet.
        For RandomDecisionMaker, it simply chooses at random a physical node that satisfies the resources requirements

        :param psn: a graph representation of the PSN
        :param vnf: a VNF to be placed on the PSN
        :return:
        """
        psn_nodes_ids = copy.deepcopy(list(psn.nodes))
        physical_node = psn.nodes[psn_nodes_ids.pop(random.randint(0, len(psn_nodes_ids) - 1))]
        while not self.resources_reqs_satisfied(physical_node, vnf) and len(psn_nodes_ids) > 0:
            physical_node = psn.nodes[psn_nodes_ids.pop(random.randint(0, len(psn_nodes_ids) - 1))]
            if len(psn_nodes_ids) == 0:
                # no physical nodes satisfy the resources requirements --> fail to accept NSPR
                return None
        return physical_node
