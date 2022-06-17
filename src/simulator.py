import reader


class Simulator:
    def __init__(self, psn_file: str, nsprs_path: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nsprs_path: either directory with the GraphML files defining the NSPRs or path to a single GraphML file
        """
        self.psn = reader.read_psn(graphml_file=psn_file)   # physical substrate network
        self.nsprs = reader.read_nsprs(nsprs_path=nsprs_path)   # network slice placement requests


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml", nsprs_path="../NSPRs/dummy_NSPR.graphml")
