import networkx as nx
from typing import Tuple, Dict


def _check_if_graphml(file: str):
    """ Checks if a file is a GraphML file (checking the extension)

    :param file: path to the file to be checked
    :raise ValueError: is case the file is not a GraphML file
    """
    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))


def _check_required_attributes(network: nx.Graph, required_node_attributes: Tuple[str, ...],
                               required_link_attributes: Tuple[str, ...], **admissible_values: tuple):
    """ Checks whether all the required attributes are present in the nodes and link of the network passed as argument

    :param network: network whose nodes and links have to be checked (if they contain all the required attributes)
    :param required_node_attributes: tuple containing all the required attributes for the network's nodes
    :param required_link_attributes: tuple containing all the required attributes for the network's links
    :param admissible_values: (optional) extra arguments where the name is an attribute name and the value is a tuple with the admissible values

    :raise AssertionError:
        - in case some nodes/links don't contain all the required parameters
        - in case some non admissible values are used for some arguments
    """
    # check nodes
    for node_id, node in network.nodes.items():
        # check that all required attributes are present in the current node
        assert all(req_attrib in node.keys() for req_attrib in required_node_attributes)
        # check that - if the admissible values for a certain attribute are passed - the value of each attribute is admissible
        for attrib, value in node.items():
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("CPUcap", "RAMcap", "reqCPU", "reqRAM"):
                assert value >= 0
    # check edges
    for node_A, node_B in list(network.edges):
        cur_link_attribs = network.edges[node_A, node_B].keys()
        cur_link_values = network.edges[node_A, node_B].values()
        cur_link_attribs_values = zip(cur_link_attribs, cur_link_values)
        # check that all required attributes are present in the current link
        assert all(attrib in cur_link_attribs for attrib in required_link_attributes)
        # check that - if the admissible values for a certain attribute are passed - the value of each attribute is admissible
        for attrib, value in cur_link_attribs_values:
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("BWcap", "reqBW"):
                assert value >= 0


def read_psn(graphml_file: str) -> nx.Graph:
    """ Reads a GraphML file containing the definition of a PSN

    :param graphml_file: GraphML file containing the definition of the PSN
    :return: a networkx.Graph representing the PSN

    :raise ValueError: if "graphml_file" is not a GraphML file
    :raise AssertionError: if some required attributes of nodes and links are missing
    """
    _check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    psn = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    _check_required_attributes(network=psn,
                               required_node_attributes=("NodeType", "CPUcap", "RAMcap"),
                               required_link_attributes=("BWcap",),
                               NodeType=("UAP", "router", "switch", "server"))
    return psn


def read_nspr(graphml_file: str) -> nx.Graph:
    _check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    nspr = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    _check_required_attributes(network=nspr,
                               required_node_attributes=("reqCPU", "reqRAM"),
                               required_link_attributes=("reqBW",))
    return nspr
