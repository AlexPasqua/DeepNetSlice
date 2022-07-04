import math

import networkx as nx

import decision_makers
import reader


class Simulator:
    """ Class implementing the network simulator

    Attributes:
        psn (nx.Graph): physical substrate network
        nsprs (dict): dict of NSPRs associated to their arrival time
        decision_maker (DecisionMaker): decision maker used to decide the next VNF to place onto the PSN
    """
    def __init__(self, psn_file: str, nsprs_path: str, decision_maker_type: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        :param decision_maker_type: type of decision maker
        """
        self.psn = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs = reader.read_nsprs(nsprs_path=nsprs_path)  # network slice placement requests
        self.decision_maker = decision_makers.decision_makers[decision_maker_type]
        self.reqBW = 0  # attribute needed in method self.compute_link_weight

    @staticmethod
    def get_cur_nvf_links(vnf_id: int, nspr: nx.Graph) -> dict:
        """ Get all the virtual links connected to a specific VNF

        :param vnf_id: ID of a VNF whose VLs have to be returned
        :param nspr: the NSPR to which the VNF belongs
        :return: dict of the VLs connected to the specified VNF
        """
        vnf_links = {}
        for extremes, vl in nspr.edges.items():
            if vnf_id in extremes:
                vnf_links[extremes] = vl
        return vnf_links

    def compute_link_weight(self, source: int, target: int, link: dict):
        """ Compute the weight of an edge between two nodes.
        If the edge satisfies the bandwidth requirement, the weight is 1, else infinity.

        This method is passed to networkx's shortest_path function as a weight function, and it's subject to networkx's API.
        It must take exactly 3 arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge.
        We need the required bandwidth to compute an edge's weight, so we save it into an attribute of the simulator (self.reqBW).

        :param source: source node in the PSN
        :param target: target node in the PSN
        :param link: dict of the link's (source - target) attributes
        :return: the weight of that link
        """
        return 1 if link['availBW'] >= self.reqBW else math.inf

    def evaluate_nspr(self, nspr: nx.Graph) -> bool:
        """ Place all the VNFs and VLs onto the physical network and update the available resources

        :param nspr: a NSPR to be evaluated (accepted and placed / reject)
        :return: True if the NSPR is accepted and placed on the PSN, else False
        """
        for vnf_id, vnf in nspr.nodes.items():
            if vnf['placed'] < 0:   # it means the VNF is not currently placed onto a physical node
                # select the physical node onto which to place the VNF
                physical_node_id, physical_node = self.decision_maker.decide_next_node(psn=self.psn, vnf=vnf)
                if physical_node_id == -1:
                    # it wasn't possible to place the VNF onto the PSN --> NSPR rejected
                    return False

                # place the VNF and update the resources availabilities of the physical node
                vnf['placed'] = physical_node_id
                physical_node['availCPU'] -= vnf['reqCPU']
                physical_node['availRAM'] -= vnf['reqRAM']

                # connect the placed VNF to the other VNFs it's supposed to be connected to
                cur_vnf_VLs = self.get_cur_nvf_links(vnf_id, nspr)    # get the VLs involving the current VNF
                for (source_vnf, target_vnf), vl in cur_vnf_VLs.items():
                    # get the physical nodes where the source and target VNFs are placed
                    source_node, target_node = nspr.nodes[source_vnf]['placed'], nspr.nodes[target_vnf]['placed']

                    # if the VL isn't placed yet and both the source and target VNFs are placed, place the VL
                    # NOTE: this 'if' and its body already take care of connected VNFs placed on the same physical node
                    if not vl['placed'] and source_node >= 0 and target_node >= 0:
                        self.reqBW = vl['reqBW']
                        psn_path = nx.shortest_path(G=self.psn, source=source_node, target=target_node,
                                                    weight=self.compute_link_weight, method='dijkstra')

                        # place the VL onto the PSN and update the resources availabilities of the physical links involved
                        # in case the VNFs are placed on the same physical node, this 'if' is not executed
                        for i in range(len(psn_path) - 1):
                            physical_link = self.psn.edges[psn_path[i], psn_path[i+1]]
                            physical_link['availBW'] -= vl['reqBW']
                        vl['placed'] = psn_path

    def restore_avail_resources(self, nspr: nx.Graph):
        """ Method called in case a NSPR is not accepted.
        Restores the resources if the PSN that had been already allocated for the rejected NSPR

        :param nspr: the rejected NSPR
        """
        for vnf_id, vnf in nspr.nodes.items():
            # restore nodes' resources availabilities
            if vnf['placed'] >= 0:
                physical_node = self.psn.nodes[vnf['placed']]
                physical_node['availCPU'] += vnf['reqCPU']
                physical_node['availRAM'] += vnf['reqRAM']
        for _, vl in nspr.edges.items():
            # restore links' resources availabilities
            if vl['placed']:
                # if vl['placed'] is not empty, it's the list of the physical nodes traversed by the link
                for i in range(len(vl['placed']) - 1):
                    physical_link = self.psn.edges[vl['placed'][i], vl['placed'][i+1]]
                    physical_link['availBW'] += vl['reqBW']

    def start(self, sim_steps: int = 100):
        """ Main cycle of the simulator

        :param sim_steps: number of simulation steps to be performed
        """
        current_nsprs = []
        for step in range(sim_steps):
            # add eventual newly arrived NSPRs to the list of NSPRs to be evaluated and skip if there are none
            current_nsprs += self.nsprs.get(step, [])
            if len(current_nsprs) == 0:
                continue

            # pop a NSPR from the list of NSPRs that arrived already
            cur_nspr = current_nsprs.pop(0)

            # accept/reject NSPR
            outcome = self.evaluate_nspr(nspr=cur_nspr)
            if not outcome:
                self.restore_avail_resources(nspr=cur_nspr)


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml",
                    nsprs_path="../NSPRs/",
                    decision_maker_type="random")
    sim.start()
