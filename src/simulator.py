import reader


class Simulator:
    def __init__(self, psn_file: str, nspr_file: str):
        """ Constructor
        :param psn_file: GraphML file containing the definition of the PSN
        :param nspr_file: GraphML file containing the definition of the NSPR
        """
        self.psn = reader.read_psn(graphml_file=psn_file)
        self.nspr = reader.read_nspr(graphml_file=nspr_file)


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml", nspr_file="../NSPRs/dummy_NSPR.graphml")
