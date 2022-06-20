import networkx as nx

import reader
import decision_makers


class Simulator:
    def __init__(self, psn_file: str, nsprs_path: str, decision_maker_type: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        :param decision_maker_type: type of decision maker
        """
        self.psn: nx.Graph = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs: list = reader.read_nsprs(nsprs_path=nsprs_path)  # network slice placement requests
        self.decision_maker = decision_makers.decision_makers[decision_maker_type]

    def evaluate_nspr(self, nspr: nx.Graph) -> bool:
        """ Place all the VNFs and VLs onto the physical network and update the available resources

        :param nspr: a NSPR to be evaluated (accepted and placed / reject)
        :return: True if the NSPR is accepted and placed on the PSN, else False
        """
        for vnf_id, vnf in nspr.nodes.items():
            if vnf['placed'] < 0:   # it means the VNF is not currently placed onto a physical node
                # select the physical node onto which to place the VNF
                physical_node = self.decision_maker.decide_next_node(psn=self.psn, vnf=vnf)
                if physical_node is None:
                    # it wasn't possible to place the VNF onto the PSN --> NSPR rejected
                    return False

                # update the resources capacities of the physical node
                # vnf['placed'] =
                physical_node['availCPU'] -= vnf['reqCPU']
                physical_node['availRAM'] -= vnf['reqRAM']

    def restore_avail_resources(self, nspr: nx.Graph):
        for vnf_id, vnf in nspr.nodes:
            # TODO: restore nodes' resources availabilities
            raise NotImplementedError
        for link_id, link in nspr.edges:
            # TODO: restore links' resources availabilities
            raise NotImplementedError

    def start(self, sim_steps: int = 100):
        for step in range(sim_steps):
            # skip step if there aren't NSPRs
            if len(self.nsprs) == 0:
                continue

            # pop a NSPR from the list of NSPRs that arrived already
            cur_nspr = self.nsprs.pop(0)

            # accept/reject NSPR
            outcome = self.evaluate_nspr(nspr=cur_nspr)
            if not outcome:
                self.restore_avail_resources(nspr=cur_nspr)

            # place all the VNFs and VLs onto the physical network and update the available resources
            for vnf_id, vnf in cur_nspr.nodes.items():
                if not vnf['placed']:
                    # select the physical node onto which to place the VNF
                    physical_node = self.decision_maker.decide_next_node(psn=self.psn, vnf=vnf)
                    if physical_node is None:
                        # TODO: do something to keep track of the non-placed NSPR
                        pass

                    # update the resources capacities of the physical node
                    physical_node['CPUcap'] -= vnf['reqCPU']
                    physical_node['RAMcap'] -= vnf['reqRAM']

            # TODO: check if new NSPRs arrived


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml",
                    nsprs_path="../NSPRs/dummy_NSPR.graphml",
                    decision_maker_type="random")
    sim.start()
