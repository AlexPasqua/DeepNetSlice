import networkx as nx

import reader


class Simulator:
    def __init__(self, psn_file: str, nsprs_path: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        """
        self.psn: nx.Graph = reader.read_psn(graphml_file=psn_file)  # physical substrate network
        self.nsprs: list = reader.read_nsprs(nsprs_path=nsprs_path)  # network slice placement requests

    def start(self, sim_steps: int = 100):
        for step in range(sim_steps):
            # pop a NSPR from the list of NSPRs that arrived already
            if len(self.nsprs) == 0:
                continue
            cur_nspr = self.nsprs.pop(0)

            # place all the VNFs and VLs onto the physical network and update the available resources:
            selected_physical_node = ...    # TODO: finish this

            # TODO: check if new NSPRs arrived


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml", nsprs_path="../NSPRs/dummy_NSPR.graphml")
    sim.start()
