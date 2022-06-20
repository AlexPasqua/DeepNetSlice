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

    def start(self, sim_steps: int = 100):
        for step in range(sim_steps):
            # pop a NSPR from the list of NSPRs that arrived already
            if len(self.nsprs) == 0:
                continue
            cur_nspr = self.nsprs.pop(0)

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
