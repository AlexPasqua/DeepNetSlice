import reader


class Simulator:
    def __init__(self, psn_file: str):
        self.psn = reader.read_psn(graphml_file=psn_file)


if __name__ == '__main__':
    sim = Simulator(psn_file="../PSNs/triangle.graphml")
