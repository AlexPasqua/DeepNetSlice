import copy
import random
import networkx as nx
from typing import Union, Tuple

from .base_decision_maker import BaseDecisionMaker


class RandomDecisionMaker(BaseDecisionMaker):
    """
    Random decision maker: when deciding a physical node where to place a VNF,
    it chooses one at random that satisfies the resources requirements
    """
    def __init__(self):
        super().__init__()

    def decide_next_node(self, psn: nx.Graph, vnf: dict) -> Tuple[int, dict]:
        """
        Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet.
        For RandomDecisionMaker, it simply chooses at random a physical node that satisfies the resources requirements

        :param psn: a graph representation of the PSN
        :param vnf: a VNF to be placed on the PSN
        :return: the ID and the physical node itself onto which to place the VNF (if not possible, returns -1, {})
        """
        psn_servers_ids = [node_id for node_id, node in psn.nodes.items() if node['NodeType'] == "server"]
        selected_id = psn_servers_ids.pop(random.randint(0, len(psn_servers_ids) - 1))
        while not self.resources_reqs_satisfied(psn.nodes[selected_id], vnf) and len(psn_servers_ids) > 0:
            selected_id = psn_servers_ids.pop(random.randint(0, len(psn_servers_ids) - 1))
            if len(psn_servers_ids) == 0:
                # no physical nodes satisfy the resources requirements --> fail to accept NSPR
                return -1, {}
        return selected_id, psn.nodes[selected_id]
