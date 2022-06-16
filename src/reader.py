import networkx as nx


def read_psn(graphml_file: str) -> nx.Graph:
    """ Reads a GraphML file containing the definition of a PSN

    :param graphml_file: GraphML file containing the definition of the PSN
    :return: a networkx.Graph representing the PSN

    :raise ValueError: if "graphml_file" is not a GraphML file
    :raise AssertionError: if some required attributes of nodes and links are missing
    """
    # check if the file passed is a GraphML file
    if not graphml_file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(graphml_file))

    # read the GraphML file and create a nx.Graph object
    psn = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    required_node_attributes = ("NodeType", "CPUcap", "RAMcap")
    required_link_attributes = ("BWcap",)
    for node_id, node in psn.nodes.items():
        assert all(attrib in node.keys() for attrib in required_node_attributes)
    for node_A, node_B in list(psn.edges):
        cur_link_attribs = psn.edges[node_A, node_B].keys()
        assert all(attrib in cur_link_attribs for attrib in required_link_attributes)

    return psn
